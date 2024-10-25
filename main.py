import sys
from logzero import logger
import argparse
import torch
from torch import optim
import numpy as np
from torch_utils.model import DualGOFiller
from torch_utils.loss_fn import bpr_loss, InfoNCE
from torch_utils.helper import convert_sp_mat_to_sp_tensor, _convert_sp_mat_to_sp_tensor, norm_adj
from utils.data_utils import Data
from utils.evaluation import find_k_largest, ranking_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../dataset', help='path to the data file')
    parser.add_argument('--NS', type=str, default='mf', help='domains in GO, mf for molecular function, bp for biological process, cc for cellular component')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--n_negs', type=int, default=1, help='number of negative samples')
    parser.add_argument('--temperature', type=float, default=0.15, help='temperature for InfoNCE')
    parser.add_argument('--device', type=str, default='cuda:0', help='device for training')
    parser.add_argument('--save_path', type=str, default='./model.pth', help='path to save the model')
    parser.add_argument('--alpha', type=float, default=0.5, help='balance parameter for the two channels')
    parser.add_argument('--beta', type=float, default=0.01, help='balance parameter for the two losses')
    parser.add_argument('--topN', type=int, default=8, help='top N for evaluation, mf for 8, bp for 25, cc for 12')
    args = parser.parse_args()

    # load data
    data = Data(args.data_path, args.NS, args.batch_size)
    norm_adj_mat = data.create_adj_mat()
    ppi_mat, protein_embedding = data.load_ppi_mat()
    go_mat, go_embedding = data.load_go_mat()

    # prepare data
    train_data = {'X_esm': protein_embedding.to(args.device), 
                  'X_pub': go_embedding.to(args.device),
                  'A_esm': norm_adj(convert_sp_mat_to_sp_tensor(ppi_mat)).to(args.device),
                  'A_pub': norm_adj(convert_sp_mat_to_sp_tensor(go_mat)).to(args.device),
                  'A_bi': _convert_sp_mat_to_sp_tensor(norm_adj_mat).to(args.device)}
    
    # define model and optimizer
    model = DualGOFiller(data.n_proteins, data.n_terms, protein_embedding.shape[1], go_embedding.shape[1], 64)
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    bestPerformance = []
    for epoch in range(args.n_epochs):
        # train
        model.train()
        optimizer.zero_grad()
        loss = 0
        for n, batch in enumerate(data.next_batch_pairwise()):
            pro_idx, pos_idx, neg_idx = batch
            ppi_pro_emb, dag_term_emb, rec_pro_emb, rec_term_emb = model(**train_data)
            rec_pro_emb, rec_pos_term_emb, rec_neg_term_emb = rec_pro_emb[pro_idx], rec_term_emb[pos_idx], rec_term_emb[neg_idx]
            batch_loss_bi = bpr_loss(rec_pro_emb, rec_pos_term_emb, rec_neg_term_emb)

            homo_pro_emb, homo_pos_term_emb, homo_neg_term_emb = ppi_pro_emb[pro_idx], dag_term_emb[pos_idx], dag_term_emb[neg_idx]
            # batch_loss = bpr_loss(rec_pro_emb + 0.5 * homo_pro_emb, rec_pos_term_emb + 0.5 * homo_pos_term_emb, rec_neg_term_emb + 0.5 * homo_neg_term_emb)
            batch_loss_homo = bpr_loss(homo_pro_emb, homo_pos_term_emb, homo_neg_term_emb)
            batch_loss = batch_loss_bi + batch_loss_homo * args.alpha
            # contrastive loss
            pro_info_loss = InfoNCE(rec_pro_emb, ppi_pro_emb, temperature=args.temperature) # mf bp 0.1
            term_info_loss = InfoNCE(rec_term_emb, dag_term_emb, temperature=args.temperature) # cc 0.15
            
            batch_loss += (pro_info_loss + term_info_loss) * args.beta
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()

        logger.info(f"Epoch {epoch+1} loss: {loss:.4f}")
        torch.save(model.state_dict(), f"{args.save_path}/{args.NS}_model_epoch_{epoch}.pth")
        torch.cuda.empty_cache()

        if epoch % 5 == 0:
            all_prot_list = [uid for uid in data.test_set]
            with torch.no_grad():
                ppi_pro_emb, dag_term_emb, rec_pro_emb, rec_term_emb = model(**train_data)
                ppi_pro_emb, rec_pro_emb = ppi_pro_emb[all_prot_list], rec_pro_emb[all_prot_list]
                # pro_emb = rec_pro_emb + 0.5 * ppi_pro_emb
                # term_emb = rec_term_emb + 0.5 * dag_term_emb
                # pred_mat = torch.matmul(pro_emb, term_emb.t()).detach().cpu().numpy()
                pred_mat1 = torch.matmul(ppi_pro_emb, dag_term_emb.t()).detach().cpu().numpy()
                pred_mat2 = torch.matmul(rec_pro_emb, rec_term_emb.t()).detach().cpu().numpy()
                pred_mat = pred_mat2 + 0.5 * pred_mat1
            rec_list = {}
            origin = {}
            for i, uid in enumerate(all_prot_list):
                test_uid_terms = data.test_set[uid]
                train_uid_terms = data.train_terms[uid]
                for terms in train_uid_terms:
                    pred_mat[i][terms] = -10e8
                origin[uid] = {}
                for terms in test_uid_terms:
                    # pred_mat[uid][terms] = -10e8
                    origin[uid][terms] = 1
                candidates = pred_mat[i, :]
                ids, scores = find_k_largest(np.max(args.topN), candidates)
                rec_list[uid] = list(zip(ids, scores))
            # result = ranking_evaluation(origin, rec_list, topN[ns])
            # epoch = int(path.split('_')[-1].split('.')[0])
            # model_name = f'DualGOFiller_{ns}_{epoch}'
            measure = ranking_evaluation(origin, rec_list, [args.topN])
            if len(bestPerformance) > 0:
                count = 0
                performance = {}
                for m in measure[1:]:
                    k, v = m.strip().split(':')
                    performance[k] = float(v)
                for k in bestPerformance[1]:
                    if k not in ['F1', 'NDCG']:
                        continue
                    if bestPerformance[1][k] > performance[k]:
                        count += 1
                    else:
                        count -= 1
                if count < 0:
                    bestPerformance[1] = performance
                    bestPerformance[0] = epoch + 1
            else:
                bestPerformance.append(epoch + 1)
                performance = {}
                for m in measure[1:]:
                    k, v = m.strip().split(':')
                    performance[k] = float(v)
                bestPerformance.append(performance)

            logger.info('-' * 120)
            logger.info(f'Real-Time Ranking Performance Top-{args.topN} Term Quality')
            measure = [m.strip() for m in measure[1:]]
            logger.info('*Current Performance*')
            measurement = ' | '.join(measure)
            logger.info(f'Epoch: {str(epoch + 1)}, {measurement}')

            bp = ''
            # for k in self.bestPerformance[1]:
            #     bp+=k+':'+str(self.bestPerformance[1][k])+' | '
            bp += 'Hit Ratio' + ':' + str(bestPerformance[1]['Hit Ratio']) + '  |  '
            bp += 'Precision' + ':' + str(bestPerformance[1]['Precision']) + '  |  '
            bp += 'Recall' + ':' + str(bestPerformance[1]['Recall']) + '  |  '
            bp += 'F1' + ':' + str(bestPerformance[1]['F1']) + ' | '
            bp += 'NDCG' + ':' + str(bestPerformance[1]['NDCG']) + '  |  '
            logger.info('*Best Performance* ')
            logger.info(f'Epoch: {str(bestPerformance[0])},  |  {bp}')
            logger.info('-' * 120)
                
            
if __name__ == '__main__':
    main()