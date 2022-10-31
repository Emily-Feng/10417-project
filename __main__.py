import torch.nn.functional as F

from model import LSTMClassifier
from utils import train_model, eval_model, load_dataset

LEARNING_RATE = 2e-5
BATCH_SIZE = 32
OUTPUT_SIZE = 2
HIDDEN_SIZE = 256
EMBEDDING_LENGTH = 300
EPOCHS = 10

def main():
    TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_dataset()

    model = LSTMClassifier(BATCH_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, vocab_size, EMBEDDING_LENGTH, word_embeddings)
    loss_fn = F.cross_entropy


    for epoch in range(EPOCHS):
        train_loss, train_acc = train_model(model, train_iter, epoch)
        val_loss, val_acc = eval_model(model, valid_iter)

        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')



if __name__ == "__main__":
    main()