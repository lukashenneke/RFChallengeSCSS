import torch

class MSE_BCE_loss(torch.nn.Module):

    def __init__(self, lambda_mse=1.0, lambda_ber=1.0):
        super().__init__()
        self.MSE = torch.nn.MSELoss()
        self.lambda_mse = lambda_mse
        self.BER = torch.nn.BCEWithLogitsLoss()
        self.lambda_ber = lambda_ber
        self.tmp_mse, self.tmp_ber = 0, 0 

    def forward(self, pred, target):
        yp, bp = pred
        yt, bt = target
        self.tmp_mse = self.MSE(yp, yt)
        self.tmp_ber = self.BER(bp, bt)
        loss = self.lambda_mse * self.tmp_mse + self.lambda_ber * self.tmp_ber
        return loss


class MSE_MSE_loss(torch.nn.Module):

    def __init__(self, lambda_mse=1.0, lambda_ber=1.0):
        super().__init__()
        self.MSE = torch.nn.MSELoss()
        self.lambda_mse = lambda_mse
        self.BER = torch.nn.MSELoss()
        self.lambda_ber = lambda_ber
        self.tmp_mse, self.tmp_ber = 0, 0 

    def forward(self, pred, target):
        yp, bp = pred
        yt, bt = target
        self.tmp_mse = self.MSE(yp, yt)
        self.tmp_ber = self.BER(bp, bt)
        loss = self.lambda_mse * self.tmp_mse + self.lambda_ber * self.tmp_ber
        return loss
