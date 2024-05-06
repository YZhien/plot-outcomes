import json
import matplotlib.pyplot as plt

with open('logSSfood.json') as f:
        with open('logSSREfood.json') as R:
                data_ss = json.load(f)
                data_sr = json.load(R)

                epochs_ss = [entry['epoch'] for entry in data_ss]
                epochs_sr = [entry['epoch'] for entry in data_ss]
                #train_losses = [entry['train']['loss'] for entry in data]
                test_losses_ss = [entry['test']['loss'] for entry in data_ss]
                test_losses_sr = [entry['test']['loss'] for entry in data_sr]
                #train_accuracies = [entry['train']['accuracy'] for entry in data]
                test_accuracies_ss = [entry['test']['accuracy'] for entry in data_ss]
                test_accuracies_sr = [entry['test']['accuracy'] for entry in data_sr][:len(epochs_ss)]
                print(str(len(test_accuracies_sr)))
                print(str(len(test_accuracies_ss)))

                #train_errors = [1 - accuracy for accuracy in train_accuracies]
                test_errors_ss = [1 - accuracy for accuracy in test_accuracies_ss]
                test_errors_sr = [1 - accuracy for accuracy in test_accuracies_sr][:len(epochs_ss)]
                print(str(len(test_errors_ss)))
                print(str(len(test_errors_sr)))

                plt.figure(figsize=(10, 5))

                plt.subplot(1, 2, 1)
                plt.plot(epochs_ss, test_losses_ss[:len(epochs_ss)], label='shakeshake',alpha=0.5)
                plt.plot(epochs_ss, test_losses_sr[:len(epochs_ss)], label='shakeshake and RE', alpha=0.5)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Test Losses vs. Epochs on food dataset')
                plt.legend()
                plt.grid(True)

                plt.subplot(1, 2, 2)
                plt.plot(epochs_ss, test_errors_ss, label='shakeshake', alpha=0.5)
                plt.plot(epochs_ss, test_errors_sr, label='shakeshake and RE', alpha=0.5)
                plt.xlabel('Epoch')
                plt.ylabel('Error')
                plt.title(' Test Errors vs. Epochs on food dataset')
                plt.legend()
                plt.grid(True)

                plt.savefig('loss_and_error_plot.png')

                plt.show()
