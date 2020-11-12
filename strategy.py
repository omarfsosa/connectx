import torch
import torch.nn as nn
import torch.optim as optim


class Strategy:
    def __init__(self, model, lr=3e-5, wd=0.001):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def eval_position(self, board, config=None):
        x = self._board_to_input(board, config)
        prediction = self.model(x)
        return prediction

    def update(self, boards, rewards):
        self.optimizer.zero_grad()
        y_pred = self.eval_position(boards)
        y_true = torch.tensor(rewards, dtype=torch.float, device=self.device)
        y_true = y_true.view(*y_pred.shape)
        error = self.loss(y_true, y_pred)
        error.backward()
        self.optimizer.step()

    def _board_to_input(self, board, config):
        """
        Takes a board (or a list of boards) and puts
        it into a tensor shaped input
        """
        if config is None:
            rows = 6
            cols = 7
        else:
            rows = config.rows
            cols = config.columns
        
        x = torch \
            .tensor(board, dtype=torch.float, device=self.device) \
            .view(-1, 1, rows, cols)
            
        return x

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Saved state_dict at: {path}.")

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()