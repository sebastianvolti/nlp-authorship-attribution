import sys
from constants import MODEL_TYPES
import data
import model

#run: python3 main.py
def main():

    datasets = data.Data()
    model_type = MODEL_TYPES["SIMPLE"]         # HERE GOES THE MODEL WE CONFIGURED IN CORSS VALIDATION
    epochs = 5                                 # HERE GOES THE EPOCHS WE CONFIGURED IN CORSS VALIDATION
    neurons = 128                              # HERE GOES THE NEURONS WE CONFIGURED IN CORSS VALIDATION
    dropout = 0.25                             # HERE GOES THE DROPOUT WE CONFIGURED IN CORSS VALIDATION
    batches_size = 256                         # HERE GOES THE BATCH_SIZE WE CONFIGURED IN CORSS VALIDATION

    m = model.Model(
        model_type=model_type,
        train_dataset=datasets.train,
        neurons=neurons,
        dropout=dropout,
        val_dataset=datasets.val,
        test_dataset=None
    )
    m.train(epochs, batches_size)
    m.eval()

if __name__ == "__main__":
    main()