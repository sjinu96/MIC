import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.autograd import Variable
import torchvision.models as models

#import sys
#sys.path.append('/home/medinfo/anaconda_home/OpenNMT')
#from onmt.decoders.decoder import DecoderBase
#from onmt.modules import MultiHeadedAttention, AverageAttention
#from onmt.modules.position_ffn import PositionwiseFeedForward
#from onmt.utils.misc import sequence_mask


class VisualFeatureExtractor(nn.Module):
    def __init__(self, model_name='densenet201', pretrained=False):
        super(VisualFeatureExtractor, self).__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.conv5_model, self.conv5_out_features, self.conv5_avg_func, \
        self.conv4_out_features, self.conv4_avg_func, \
        self.conv3_out_features, self.conv3_avg_func = self.__get_model()
        self.activation = nn.ReLU()
        
        ## Fully Connected Layer
        self.conv5_fc = nn.Linear(in_features=self.conv5_out_features, out_features=512) # 512 OOM
        self.conv4_fc = nn.Linear(in_features=self.conv4_out_features, out_features=512)
        self.conv3_fc = nn.Linear(in_features=self.conv3_out_features, out_features=512)
        #self.conv2_fc = nn.Linear(in_features=self.conv2_out_features, out_features=128)
        
        #self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14)) # why 14?
        #self.fc = nn.Linear(in_features=2048, out_features=512)
        
        self.__init_weight()
        
    # FC weight, bias for each Conv Layer
    def __init_weight(self):
        self.conv5_fc.weight.data.uniform_(-0.1, 0.1)
        self.conv5_fc.bias.data.fill_(0)
        self.conv4_fc.weight.data.uniform_(-0.1, 0.1)
        self.conv4_fc.bias.data.fill_(0)
        self.conv3_fc.weight.data.uniform_(-0.1, 0.1)
        self.conv3_fc.bias.data.fill_(0)
        #self.conv2_fc.weight.data.uniform_(-0.1, 0.1)
        #self.conv2_fc.bias.data.fill_(0)

    def __get_model(self):
        conv5_model, conv5_out_features, conv5_avg_func = None, None, None
        conv4_out_features, conv3_out_features = None, None
        conv4_avg_func, conv3_avg_func = None, None
        if self.model_name == 'resnet152':
            resnet = models.resnet152(pretrained=self.pretrained)
            
            ### ~ Stage5 Model
            conv5_modules = list(resnet.children())[:-2]        # ~Conv5
            conv5_model = nn.Sequential(*conv5_modules)         # builds a sequential model based on it that excludes the final two modules (e.g., the one that does average pooling and the fully connected one)
            for param in conv5_model.parameters():
                param.requires_grad = False
            
            conv5_out_features = resnet.fc.in_features # output nodes of the last layer of ResNet-152
            #conv5_avg_func = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

            ### ~ Stage4 Model
            conv4_out_features = 1024   #resnet.layer3.bn3.num_features
            #conv4_avg_func = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)

            ### ~ Stage3 Model
            conv3_out_features = 512
            #conv3_avg_func = torch.nn.AvgPool2d(kernel_size=28, stride=1, padding=0)

            ### ~ Stage2 Model
            #conv2_out_features = 256
            #conv2_avg_func = torch.nn.AvgPool2d(kernel_size=56, stride=1, padding=0)
        elif self.model_name == 'densenet201':
            densenet = models.densenet201(pretrained=self.pretrained)
            modules = list(densenet.features)
            model = nn.Sequential(*modules)
            func = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            out_features = densenet.classifier.in_features
        
        return conv5_model, conv5_out_features, conv5_avg_func, conv4_out_features, conv4_avg_func, \
               conv3_out_features, conv3_avg_func #, conv2_out_features, conv2_avg_func

    def forward(self, images):
        """
        :param images:
        :return:
        """
        conv5_visual_features = self.conv5_model(images)
        #images:[B,3,224,224], conv5_visual_features: [B,2048,7,7]
        conv5_visual_features = torch.flatten(conv5_visual_features, 2, 3) # ([B,2048,7*7])
        conv5_visual_features = conv5_visual_features.permute(2,0,1)       # ([7*7, B, 2048])
        conv5_visual_features = self.conv5_fc(conv5_visual_features)             # ([7*7, B, 512])
        #print("conv5_visual_features.shape:", conv5_visual_features.shape)

        conv4_visual_features = self.conv5_model[:-1](images)
        # conv4_visual_features: [B,1024,14,14]
        conv4_visual_features = torch.flatten(conv4_visual_features, 2, 3) # ([B,1024,14*14])
        conv4_visual_features = conv4_visual_features.permute(2,0,1)       # ([14*14, B, 1024])
        conv4_visual_features = self.conv4_fc(conv4_visual_features)             # ([14*14, B, 512])

        conv3_visual_features = self.conv5_model[:-2](images)    
        # conv3_visual_features: [B, 512, 28, 28]
        conv3_visual_features = torch.flatten(conv3_visual_features, 2, 3) # ([B,512,28*28])
        conv3_visual_features = conv3_visual_features.permute(2,0,1)       # ([28*28, B, 512])
        conv3_visual_features = self.conv3_fc(conv3_visual_features)             # ([28*28, B, 512])
        #print("conv3_visual_features.shape:", conv3_visual_features.shape)

        """conv2_visual_features = self.conv5_model[:-3](images)    
        # conv2_visual_features: [B, 256, 56, 56]
        conv2_visual_features = torch.flatten(conv2_visual_features, 2, 3) # ([B,128,56*56])
        conv2_visual_features = conv2_visual_features.permute(2,0,1)       # ([56*56, B, 256])
        conv2_visual_features = self.conv2_fc(conv2_visual_features)             # ([56*56, B, 512])
        #print("conv2_visual_features.shape:", conv2_visual_features.shape)"""
        
        return conv5_visual_features, conv4_visual_features, conv3_visual_features  #, conv2_visual_features


class MLC(nn.Module):
    def __init__(self,
                 classes=156,
                 sementic_features_dim=512,
                 fc_in_features=2048,
                 k=10):
        super(MLC, self).__init__()
        self.classifier = nn.Linear(in_features=fc_in_features, out_features=classes)   # 2048, 210
        self.embed = nn.Embedding(classes, sementic_features_dim)
        self.k = k
        self.softmax = nn.Softmax()
        self.__init_weight()

    def __init_weight(self):
        self.classifier.weight.data.uniform_(-0.1, 0.1)
        self.classifier.bias.data.fill_(0)

    def forward(self, avg_features):
        tags = self.softmax(self.classifier(avg_features))
        semantic_features = self.embed(torch.topk(tags, self.k)[1])
        return tags, semantic_features                          # torch.Size([4, 156]), torch.Size([4, 10, 512])


class CoAttention(nn.Module):
    def __init__(self,
                 version='v1',
                 embed_size=512,
                 hidden_size=512,
                 visual_size=2048,
                 k=10,
                 momentum=0.1):
        super(CoAttention, self).__init__()
        self.version = version
        # Visual Features
        self.W_v = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_v_h = nn.Linear(in_features=hidden_size, out_features=visual_size)
        self.bn_v_h = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_v_att = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v_att = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)
        # Semantic Features
        self.W_a = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a = nn.BatchNorm1d(num_features=k, momentum=momentum)

        self.W_a_h = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_h = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_a_att = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_att = nn.BatchNorm1d(num_features=k, momentum=momentum)
        # Fully Connected Layer
        # self.W_fc = nn.Linear(in_features=visual_size, out_features=embed_size)  # for v3
        self.W_fc = nn.Linear(in_features=visual_size + hidden_size, out_features=embed_size)
        self.bn_fc = nn.BatchNorm1d(num_features=embed_size, momentum=momentum)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.__init_weight()

    def __init_weight(self):
        self.W_v.weight.data.uniform_(-0.1, 0.1)
        self.W_v.bias.data.fill_(0)

        self.W_v_h.weight.data.uniform_(-0.1, 0.1)
        self.W_v_h.bias.data.fill_(0)

        self.W_v_att.weight.data.uniform_(-0.1, 0.1)
        self.W_v_att.bias.data.fill_(0)

        self.W_a.weight.data.uniform_(-0.1, 0.1)
        self.W_a.bias.data.fill_(0)

        self.W_a_h.weight.data.uniform_(-0.1, 0.1)
        self.W_a_h.bias.data.fill_(0)

        self.W_a_att.weight.data.uniform_(-0.1, 0.1)
        self.W_a_att.bias.data.fill_(0)

        self.W_fc.weight.data.uniform_(-0.1, 0.1)
        self.W_fc.bias.data.fill_(0)

    def forward(self, avg_features, semantic_features, h_sent):
        if self.version == 'v1':
            return self.v1(avg_features, semantic_features, h_sent)
        elif self.version == 'v2':
            return self.v2(avg_features, semantic_features, h_sent)
        elif self.version == 'v3':
            return self.v3(avg_features, semantic_features, h_sent)
        elif self.version == 'v4':
            return self.v4(avg_features, semantic_features, h_sent)
        elif self.version == 'v5':
            return self.v5(avg_features, semantic_features, h_sent)

    def v1(self, avg_features, semantic_features, h_sent) -> object:
        """
        only training
        :rtype: object
        """
        # Visual
        W_v = self.bn_v(self.W_v(avg_features))                         # torch.Size([4, 2048])
        W_v_h = self.bn_v_h(self.W_v_h(h_sent.squeeze(1)))              # torch.Size([4, 2048]) / h_sent는 [4,1,512] 인데 squueze후엔 [4,512]

        alpha_v = self.softmax(self.bn_v_att(self.W_v_att(self.tanh(W_v + W_v_h))))     # torch.Size([4, 2048])
        v_att = torch.mul(alpha_v, avg_features)                                        # torch.Size([4, 2048])
        # Semantic
        W_a = self.bn_a(self.W_a(semantic_features))                    # torch.Size([4, 10, 512])
        W_a_h = self.bn_a_h(self.W_a_h(h_sent))                         # torch.Size([4, 1, 512]) h_sent size와 동일

        alpha_a = self.softmax(self.bn_a_att(self.W_a_att(self.tanh(torch.add(W_a_h, W_a)))))   # torch.Size([4, 10, 512])
        a_att = torch.mul(alpha_a, semantic_features).sum(1)                                    # torch.Size([4, 512])
        # visual + semantic
        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))               # torch.Size([4, 512])

        return ctx, alpha_v, alpha_a

    def v2(self, avg_features, semantic_features, h_sent) -> object:
        """
        no bn
        :rtype: object
        """
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))

        alpha_v = self.softmax(self.W_v_att(self.tanh(W_v + W_v_h)))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v3(self, avg_features, semantic_features, h_sent) -> object:
        """

        :rtype: object
        """
        W_v = self.bn_v(self.W_v(avg_features))
        W_v_h = self.bn_v_h(self.W_v_h(h_sent.squeeze(1)))

        alpha_v = self.softmax(self.W_v_att(self.tanh(W_v + W_v_h)))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.bn_a_h(self.W_a_h(h_sent))
        W_a = self.bn_a(self.W_a(semantic_features))
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v4(self, avg_features, semantic_features, h_sent):
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))

        alpha_v = self.softmax(self.W_v_att(self.tanh(torch.add(W_v, W_v_h))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v5(self, avg_features, semantic_features, h_sent):
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))

        alpha_v = self.softmax(self.W_v_att(self.tanh(self.bn_v(torch.add(W_v, W_v_h)))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(self.bn_a(torch.add(W_a_h, W_a)))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

class ScaledDotAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ScaledDotAttention, self).__init__()

        self.hidden_size = hidden_size

        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.scaling_factor = torch.rsqrt(torch.tensor(self.hidden_size, dtype= torch.float))

    def forward(self, queries, keys, values):
        """The forward pass of the scaled dot attention mechanism.
        Arguments:
            queries: The current decoder hidden state, 2D or 3D tensor. (batch_size x (k) x hidden_size)
            keys: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
            values: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
        Returns:
            context: weighted average of the values (batch_size x k x hidden_size)
            attention_weights: Normalized attention weights for each encoder hidden state. (batch_size x seq_len x 1)
            The output must be a softmax weighting over the seq_len annotations.
        """

        # ------------
        # FILL THIS IN
        # ------------
        batch_size = queries.shape[0]
        #print("queries.shape:", queries.shape)
        #print("keys.shape:", keys.shape)
        #print("values.shape:", values.shape)
        q = self.Q(queries.view(batch_size, -1, queries.shape[-1]))
        k = self.K(keys)
        v = self.V(values)
        unnormalized_attention = k@q.transpose(2,1)*self.scaling_factor
        attention_weights = self.softmax(unnormalized_attention)
        context = attention_weights.transpose(2,1)@v
        return context, attention_weights
        

class CausalScaledDotAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CausalScaledDotAttention, self).__init__()

        self.hidden_size = hidden_size
        self.neg_inf = torch.tensor(-1e7)

        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.scaling_factor = torch.rsqrt(torch.tensor(self.hidden_size, dtype= torch.float))

    def forward(self, queries, keys, values):
        """The forward pass of the scaled dot attention mechanism.
        Arguments:
            queries: The current decoder hidden state, 2D or 3D tensor. (batch_size x (k) x hidden_size)
            keys: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
            values: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
        Returns:
            context: weighted average of the values (batch_size x k x hidden_size)
            attention_weights: Normalized attention weights for each encoder hidden state. (batch_size x seq_len x 1)
            The output must be a softmax weighting over the seq_len annotations.
        """

        # ------------
        # FILL THIS IN
        # ------------
        batch_size = queries.shape[0]
        q = self.Q(queries.view(batch_size, -1, queries.shape[-1]))
        k = self.K(keys)
        v = self.V(values)
        unnormalized_attention = k@q.transpose(2,1)*self.scaling_factor
        mask = ~torch.triu(unnormalized_attention).bool()
        attention_weights = self.softmax(unnormalized_attention.masked_fill(mask, self.neg_inf))
        context = attention_weights.transpose(2,1)@v
        return context, attention_weights



class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, dropout):
        super(TransformerDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)        
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        self.self_attentions = nn.ModuleList([nn.ModuleList([CausalScaledDotAttention(
                                    hidden_size=hidden_size, 
                                 ) for i in range(self.num_heads)]) for j in range(self.num_layers)])
        self.encoder_attentions = nn.ModuleList([nn.ModuleList([ScaledDotAttention(
                                    hidden_size=hidden_size, 
                                 ) for i in range(self.num_heads)]) for j in range(self.num_layers)])
        self.attention_mlps = nn.ModuleList([nn.Sequential(
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                 ) for i in range(self.num_layers)])

        self.linear_after_causal = nn.ModuleList([nn.Linear(self.num_heads*hidden_size, hidden_size) for j in range(self.num_layers)])
        self.linear_after_scaled = nn.ModuleList([nn.Linear(self.num_heads*hidden_size, hidden_size) for j in range(self.num_layers)])

        self.out = nn.Linear(hidden_size, vocab_size)

        self.positional_encodings = self.create_positional_encodings()

        self.dropout = nn.Dropout(p=dropout)

        self.layernorms1 = nn.ModuleList([nn.LayerNorm([self.hidden_size]) for i in range(self.num_layers)])
        self.layernorms2 = nn.ModuleList([nn.LayerNorm([self.hidden_size]) for i in range(self.num_layers)])
        self.layernorms3 = nn.ModuleList([nn.LayerNorm([self.hidden_size]) for i in range(self.num_layers)])

    def forward(self, inputs, annotations):
        """Forward pass of the attention-based decoder RNN.
        Arguments:
            inputs: Input token indexes across a batch for all the time step. (batch_size x decoder_seq_len)
            annotations: The encoder hidden states for each step of the input.
                         sequence. (batch_size x seq_len x hidden_size)
            hidden_init: Not used in the transformer decoder
        Returns:
            output: Un-normalized scores for each token in the vocabulary, across a batch for all the decoding time steps. (batch_size x decoder_seq_len x vocab_size)
            attentions: The stacked attention weights applied to the encoder annotations (batch_size x encoder_seq_len x decoder_seq_len)
        """
        
        batch_size, seq_len = inputs.size()
        embed = self.embedding(inputs)  # batch_size x seq_len x hidden_size

        # THIS LINE WAS ADDED AS A CORRECTION. 
        embed = embed + self.positional_encodings[:seq_len]
        embed = self.dropout(embed)

        encoder_attention_weights_list = []
        self_attention_weights_list = []
        contexts = embed

        

        for i in range(self.num_layers):
          # ------------
          # FILL THIS IN - START
          # ------------
            concat_causal = torch.empty((batch_size, seq_len, 0), device='cuda:0')
            concat_scaled = torch.empty((batch_size, seq_len, 0), device='cuda:0')
            for j in range(self.num_heads):
                new_contexts, self_attention_weights = self.self_attentions[i][j](contexts, contexts, contexts)  # batch_size x seq_len x hidden_size
                concat_causal = torch.cat((concat_causal, new_contexts), axis=2)

            new_contexts = self.linear_after_causal[i](concat_causal) #batch_size x seq_len x hidden_size*num_heads -----> batch_size x seq_len x hidden_size
            new_contexts = self.dropout(new_contexts) #dropout
            residual_contexts = self.layernorms1[i](contexts + new_contexts) #add and norm

            for j in range(self.num_heads):
                #print("residual_contexts.shape:", residual_contexts.shape)
                #print("annotations.shape:", annotations.shape)
                new_contexts, encoder_attention_weights = self.encoder_attentions[i][j](residual_contexts, annotations, annotations) # batch_size x seq_len x hidden_size
                concat_scaled = torch.cat((concat_scaled, new_contexts), axis=2)
            
            new_contexts = self.linear_after_scaled[i](concat_scaled) #batch_size x seq_len x hidden_size*num_heads -----> batch_size x seq_len x hidden_size
            new_contexts = self.dropout(new_contexts) #dropout
            residual_contexts = self.layernorms2[i](residual_contexts + new_contexts) #add and norm

            new_contexts = self.attention_mlps[i](residual_contexts)
            new_contexts = self.dropout(new_contexts) #dropout
            contexts = self.layernorms3[i](residual_contexts + new_contexts) #add and norm
          # ------------
          # FILL THIS IN - END
          # ------------
          
            encoder_attention_weights_list.append(encoder_attention_weights)
            self_attention_weights_list.append(self_attention_weights)
          
        output = self.out(contexts)
        encoder_attention_weights = torch.stack(encoder_attention_weights_list)
        self_attention_weights = torch.stack(self_attention_weights_list)
        
        return output, (encoder_attention_weights, self_attention_weights)

    def create_positional_encodings(self, max_seq_len=1000):
        """Creates positional encodings for the inputs.
      Arguments:
          max_seq_len: a number larger than the maximum string length we expect to encounter during training
      Returns:
          pos_encodings: (max_seq_len, hidden_dim) Positional encodings for a sequence with length max_seq_len. 
      """
        pos_indices = torch.arange(max_seq_len)[..., None]
        dim_indices = torch.arange(self.hidden_size//2)[None, ...]
        exponents = (2*dim_indices).float()/(self.hidden_size)
        trig_args = pos_indices / (10000**exponents)
        sin_terms = torch.sin(trig_args)
        cos_terms = torch.cos(trig_args)

        pos_encodings = torch.zeros((max_seq_len, self.hidden_size))
        pos_encodings[:, 0::2] = sin_terms
        pos_encodings[:, 1::2] = cos_terms
        pos_encodings = pos_encodings.cuda()

        return pos_encodings


if __name__ == '__main__':
    import torchvision.transforms as transforms

    import warnings
    warnings.filterwarnings("ignore")
#
    extractor = VisualFeatureExtractor(model_name='resnet152')
    mlc = MLC(fc_in_features=extractor.out_features)
    co_att = CoAttention(visual_size=extractor.out_features)
    #src_emb = build_src_emb(model_opt, fields)
    #decoder, _ = build_decoder_with_embeddings(
    #    model_opt, fields, share_embeddings=True, src_emb=src_emb
    #)
    #return onmt.models.LanguageModel(decoder=decoder)
    #transformer = TransformerLMDecoder(num_layers=6, d_model=, heads=8, d_ff=2048, copy_attn=True, self_attn_type=, dropout=0.1, attention_dropout=0.1, embeddings=, max_relative_positions, aan_useffn=)
    #sent_lstm = SentenceLSTM()
    #word_lstm = WordLSTM(embed_size=512, hidden_size=512, vocab_size=100, num_layers=1)

    transformer = TransformerDecoder(vocab_size=1807, hidden_size=512, num_layers=6, num_heads=8, dropout=0.1)
    
    images = torch.randn((4, 3, 224, 224))
    captions = torch.ones((4, 10)).long()
    hidden_state = torch.randn((4, 1, 512))

    # # image_file = '../data/images/CXR2814_IM-1239-1001.png'
#     # # images = Image.open(image_file).convert('RGB')
#     # # captions = torch.ones((1, 10)).long()
#     # # hidden_state = torch.randn((10, 512))
# #
# norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.TenCrop(224),
#     transforms.Lambda(lambda crops: torch.stack([norm(transforms.ToTensor()(crop)) for crop in crops])),
# ])

# images = transform(images)
# images.unsqueeze_(0)
#
# # bs, ncrops, c, h, w = images.size()
# # images = images.view(-1, c, h, w)
#
    print("images:{}".format(images.shape))
    print("captions:{}".format(captions.shape))
    print("hidden_states:{}".format(hidden_state.shape))

    visual_features, avg_features = extractor.forward(images)

    print("visual_features:{}".format(visual_features.shape))
    print("avg features:{}".format(avg_features.shape))

    tags, semantic_features = mlc.forward(avg_features)

    print("tags:{}".format(tags.shape))
    print("semantic_features:{}".format(semantic_features.shape))

    ctx, alpht_v, alpht_a = co_att.forward(avg_features, semantic_features, hidden_state)

    print("ctx:{}".format(ctx.shape))
    print("alpht_v:{}".format(alpht_v.shape))
    print("alpht_a:{}".format(alpht_a.shape))

    topic, p_stop, hidden_state, states = transformer.forward(captions, hidden_state)
    # p_stop_avg = p_stop.view(bs, ncrops, -1).mean(1)

    print("Topic:{}".format(topic.shape))
    print("P_STOP:{}".format(p_stop.shape))
    # print("P_stop_avg:{}".format(p_stop_avg.shape))

    words = word_lstm.forward(topic, captions)
    print("words:{}".format(words.shape))

    cam = torch.mul(visual_features, alpht_v.view(alpht_v.shape[0], alpht_v.shape[1], 1, 1)).sum(1)
    cam.squeeze_()
    cam = cam.cpu().data.numpy()
    for i in range(cam.shape[0]):
        heatmap = cam[i]
        heatmap = heatmap / np.max(heatmap)
        print(heatmap.shape)
