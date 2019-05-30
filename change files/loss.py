import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
	longTensor = torch.cuda.LongTensor
	floatTensor = torch.cuda.FloatTensor

else:
	longTensor = torch.LongTensor
	floatTensor = torch.FloatTensor




class loss_function_new(nn.Module):
	def __init__(self):
		super(loss_function_new , self).__init__()


	def forward(self , pos_h_batch, pos_t_batch, pos_r_batch, pos_time_batch, neg_h_batch, neg_t_batch, neg_r_batch, neg_time_batch, model, ent_embeddings, L1_flag):
	    h_e = ent_embeddings[pos_h_batch]
	    t_e = ent_embeddings[pos_t_batch]
	    test_r_batch = autograd.Variable(longTensor(pos_r_batch))
	    test_time_batch = autograd.Variable(longTensor(pos_time_batch))
	    rseq_e = model.get_rseq(test_r_batch, test_time_batch).data.cpu().detach().numpy()
	    c_t_e = h_e + rseq_e
	    c_h_e = t_e - rseq_e
	    if L1_flag == True:
	        dist = pairwise_distances(c_t_e, ent_embeddings, metric='manhattan')
	    else:
	        dist = pairwise_distances(c_t_e, ent_embeddings, metric='euclidean')
	    for i in range(len(dist)):
	        for j in range(len((dist[0]))):
	            dist[i][j] = 1/dist[i][j]*100
	    y = torch.zeros(len(pos_h_batch), len(dist[0]))
	    y[range(y.shape[0]),pos_t_batch] = 1
	    pred = torch.sigmoid(torch.tensor(dist))
	    pred = pred.float().cuda()
	    y = y.cuda()
	    loss1 = F.binary_cross_entropy(pred,y)
	    # exit()



	    h_e = ent_embeddings[neg_h_batch]
	    t_e = ent_embeddings[neg_t_batch]
	    test_r_batch = autograd.Variable(longTensor(neg_r_batch))
	    test_time_batch = autograd.Variable(longTensor(neg_time_batch))
	    rseq_e = model.get_rseq(test_r_batch, test_time_batch).data.cpu().detach().numpy()
	    c_t_e = h_e + rseq_e
	    c_h_e = t_e - rseq_e
	    if L1_flag == True:
	        dist = pairwise_distances(c_t_e, ent_embeddings, metric='manhattan')
	    else:
	        dist = pairwise_distances(c_t_e, ent_embeddings, metric='euclidean')
	    for i in range(len(dist)):
	        for j in range(len((dist[0]))):
	            dist[i][j] = 1/dist[i][j]*100
	    y = torch.zeros(len(neg_h_batch), len(dist[0]))
	    y[range(y.shape[0]),pos_t_batch] = 0
	    pred = torch.sigmoid(torch.tensor(dist))
	    pred = pred.float().cuda()
	    y = y.cuda()
	    loss2 = F.binary_cross_entropy(pred,y)






	    return loss1 + loss2


class marginLoss(nn.Module):
	def __init__(self):
		super(marginLoss, self).__init__()

	def forward(self, pos, neg, margin):
		zero_tensor = floatTensor(pos.size())
		zero_tensor.zero_()
		zero_tensor = autograd.Variable(zero_tensor)
		return torch.sum(torch.max(pos - neg + margin, zero_tensor))

def orthogonalLoss(rel_embeddings, norm_embeddings):
	return torch.sum(torch.sum(norm_embeddings * rel_embeddings, dim=1, keepdim=True) ** 2 / torch.sum(rel_embeddings ** 2, dim=1, keepdim=True))

def normLoss(embeddings, dim=1):
	norm = torch.sum(embeddings ** 2, dim=dim, keepdim=True)
	return torch.sum(torch.max(norm - autograd.Variable(floatTensor([1.0])), autograd.Variable(floatTensor([0.0]))))

def regulLoss(embeddings):
	return torch.mean(embeddings ** 2)

class binaryCrossLoss(nn.Module):
	def __init__(self):
		super(binaryCrossLoss, self).__init__()

	def forward(self, cat_pos_neg ,labels):
		# pos_labels = floatTensor(pos.shape[0])
		# nn.init.ones_(pos_labels)
		# neg_labels = floatTensor(neg.shape[0])
		# nn.init.zeros_(neg_labels)
		# labels = torch.cat((pos_labels, neg_labels))
		# print(type(labels))
		# print(type(cat_pos_neg))
		# exit()



		return F.binary_cross_entropy_with_logits(cat_pos_neg, labels.cuda())
