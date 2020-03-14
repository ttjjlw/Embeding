import torch


class COBW(torch.nn.Module):
    def __init__(self, embedding_dim, vocab_size, neg_model=True):
        super(COBW, self).__init__()
        self.neg_model = neg_model
        if not self.neg_model:
            self.embedding_matrix = torch.nn.Embedding(
                vocab_size*2-1, embedding_dim)
            torch.nn.init.xavier_uniform_(self.embedding_matrix.weight.data)
        else:
            self.v_embedding_matrix = torch.nn.Embedding(vocab_size,
                                                         embedding_dim)

            torch.nn.init.xavier_uniform_(self.v_embedding_matrix.weight.data)

            self.u_embedding_matrix = torch.nn.Embedding(vocab_size,
                                                         embedding_dim)
            self.u_embedding_matrix.weight.data.uniform_(0, 0)
 
    def forward(self, pos_v, pos_u, neg_v, neg_u):
        if not self.neg_model:
            pos_v_em = None
            for i in pos_v:
                i = self.embedding_matrix(i)
                i = torch.mean(i, dim=0, keepdim=True)
                if pos_v_em is None:
                    pos_v_em = i
                else:
                    pos_v_em = torch.cat((pos_v_em, i), dim=0)
            pos_v = pos_v_em

            pos_u = self.embedding_matrix(pos_u)

            neg_v_em = None
            for i in neg_v:
                i = self.embedding_matrix(i)
                i = torch.mean(i, dim=0, keepdim=True)
                if neg_v_em is None:
                    neg_v_em = i
                else:
                    neg_v_em = torch.cat((neg_v_em, i), dim=0)
            neg_v = neg_v_em

            neg_u = self.embedding_matrix(neg_u)
        else:
            pos_v_em = None
            for i in pos_v:
                i = self.v_embedding_matrix(i)
                i = torch.mean(i, dim=0, keepdim=True)
            if pos_v_em is None:
                pos_v_em = i
            else:
                pos_v_em = torch.cat((pos_v_em, i), dim=0)
            pos_v = pos_v_em

            pos_u = self.u_embedding_matrix(pos_u)

            neg_v_em = None
            for i in neg_v:
                i = self.v_embedding_matrix(i)
                i = torch.mean(i, dim=0, keepdim=True)
            if neg_v_em is None:
                neg_v_em = i
            else:
                neg_v_em = torch.cat((neg_v_em, i), dim=0)
            neg_v = neg_v_em

            neg_u = self.u_embedding_matrix(neg_u)
        
        pos_z = torch.sum(pos_v.mul(pos_u), dim = 1,keepdim=True)
        neg_z = torch.sum(neg_v.mul(neg_u), dim = 1,keepdim=True)

        pos_a = torch.nn.functional.logsigmoid(pos_z)
        neg_a = torch.nn.functional.logsigmoid(-1 * neg_z)

        pos_loss = torch.sum(pos_a)
        neg_loss = torch.sum(neg_a)

        loss = -1 * (pos_loss + neg_loss)
        return loss
