import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model_decoding import BrainTranslator
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from data import ZuCo_dataset
import pickle
from transformers import BertTokenizer, BertLMHeadModel, BertConfig

class PretrainBrainTranslator(BrainTranslator):
    def __init__(self, *args, **kwargs):
        super(PretrainBrainTranslator, self).__init__(*args, **kwargs)
        self.fcpre = nn.Linear(1024, self.in_feature)

    def pretrain_forward(self, input_embeddings_batch, input_masks_invert):
        """Forward pass for pretraining. Masked positions should be reconstructed."""
        # Get encoded embeddings from the additional encoder
        encoded_embedding = self.addin_forward(input_embeddings_batch, input_masks_invert)
        # Project back to input feature space for reconstruction
        print(encoded_embedding.shape)
        reconstructed = self.fcpre(encoded_embedding)  # Shape: (batch_size, seq_len, in_feature)

        return reconstructed

def create_masked_input(input_embeddings, mask_ratio=0.15):
    """Randomly mask some elements in the input batch."""
    batch_size, seq_len, feature_dim = input_embeddings.size()
    mask = torch.rand(batch_size, seq_len) < mask_ratio  # Binary mask (1 = masked)
    masked_embeddings = input_embeddings.clone()
    masked_embeddings[mask.unsqueeze(-1).expand_as(masked_embeddings)] = 0.0  # Zero-out masked positions
    return masked_embeddings, mask

# Instantiate model, optimizer, and loss function
model = PretrainBrainTranslator(pretrained_layers=None, in_feature=840)
optimizer = optim.Adam(model.additional_encoder.parameters(), lr=1e-4)
criterion = nn.MSELoss()  # Reconstruction loss

# Dummy input data (batch_size=32, seq_len=100, feature_dim=840)
input_embeddings = torch.randn(32, 100, 840)
input_masks_invert = torch.zeros(32, 100).bool()  # No mask in this example


whole_dataset_dict = []
dataset_path_task1 = './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle' 
with open(dataset_path_task1, 'rb') as handle:
    whole_dataset_dict.append(pickle.load(handle))

'''set up tokenizer'''
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


train_set = ZuCo_dataset(whole_dataset_dict, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)


# Pretraining loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    
    # Generate masked input and corresponding mask
    masked_embeddings, mask = create_masked_input(input_embeddings)
    
    # Forward pass
    print('masked embeddings', masked_embeddings.shape)
    reconstructed = model.pretrain_forward(masked_embeddings, input_masks_invert)

    # Compute loss only on masked positions
    loss = criterion(reconstructed[mask], input_embeddings[mask])

    # Backpropagation and optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
