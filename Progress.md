9/18/2024
Set up the github account. added all the changes as done for a Vanilla Stochastic Neural Network for Iris dataset. 


10/30/2024
2 problems:
1. decoder:
        log images in mlflow, x and x_recon to check if it is generating avg image.
        
2. kl term:
        beta-rate scheduler start from higher scalar 



Ablation Study:

1. Entropy Term --> MLFLOW
                        NAME:  abundant-fish-104 
                        RUN ID: 6998d8d04789424c8bc6e141889e2627

1. FIM Term --> MLFLOW
                        NAME:  auspicious-skunk-671 
                        RUN ID: b359fabbb0f14ece96bcc26d5511de9e

1. Freeze Decoder --> MLFLOW
                        NAME: zealous-gnu-254 
                        RUN ID: dfb6769bfc4541adbb6a4fea6f77ec17