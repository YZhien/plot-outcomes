import json
import matplotlib.pyplot as plt

with open('log.json') as f:
        data = json.load(f)

        epochs = [entry['epoch'] for entry in data]
        train_losses = [entry['train']['loss'] for entry in data]
        test_losses = [entry['test']['loss'] for entry in data]
        train_accuracies = [entry['train']['accuracy'] for entry in data]
        test_accuracies = [entry['test']['accuracy'] for entry in data]

        train_errors = [1 - accuracy for accuracy in train_accuracies]
        test_errors = [1 - accuracy for accuracy in test_accuracies]

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Test Losses vs. Epochs')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_errors, label='Train Error')
        plt.plot(epochs, test_errors, label='Test Error')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Train and Test Errors vs. Epochs')
        plt.legend()
        plt.grid(True)

        plt.savefig('loss_and_error_plot.png')

        plt.show()
