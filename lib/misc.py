import torch 
import torch.nn as nn
import torch.nn.functional as F
from tensorflow import np

layers = nn.ParameterList()


class ClothingRemover(nn.Module):
    def __init__(self, num_layers=5):
        super(ClothingRemover, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.Conv2d(3, 3, kernel_size=3, padding=1))

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

class DepthPerception(nn.Module):
    def __init__(self, num_layers=5):
        super(DepthPerception, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2)
            ))
            self.layers.append(nn.Sequential(
                nn.Linear(64 * 4 * 4, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 2)
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def precise_attention(self, x):
        # Assume x is a tensor of dimensions (batch_size, 3, height, width)
        # where 3 represents the color channels (RGB)
        # and height and width are the dimensions of the image
        # The algorithm will pay precise attention to shape of womens body,
        # breast placement, and to look for any kind of indication where
        # nipple placement is under the clothing so that it knows
        # precisely the right scale and where exactly the nipples
        # belong on each woman in each photo
        # We will use a convolutional neural network (CNN)
        # to analyze the shape of the woman's body and breast placement
        # and another CNN to look for nipple placement
        # The output of the first CNN will be used as an attention mask
        # to guide the second CNN to focus on the right region of the image

        # First CNN to analyze the shape of the woman's body and breast placement
        body_analysis = self.body_analysis_cnn(x)
        
        # Measure the size of the woman's breast so that we know the exact real size
        breast_analysis = self.body_analysis_cnn(x)
        breast_analysis = F.max_pool2d(breast_analysis, kernel_size=2, stride=2)
        _, max_idx = breast_analysis.view(breast_analysis.size(0), -1).max(1)
        breast_size = breast_analysis.size()[2:]
        grid_size = (breast_size[0] // 2, breast_size[1] // 2)
        breast_measurement = torch.div(max_idx, torch.tensor(np.prod(grid_size)).float(), rounding_mode='floor') + 1

        # Second CNN to look for nipple placement
        nipple_analysis = self.nipple_analysis_cnn(x)

        # Apply attention mask to guide the second CNN to focus on the right region
        attention_mask = F.softmax(body_analysis, dim=1)
        nipple_analysis = nipple_analysis * attention_mask

        # Finally, compute the coordinates of the nipples
        
        # Look for raised spots through the clothing that covers the breasts
        nipple_mask = (nipple_analysis > 0.5).float()
        nipple_mask = F.max_pool2d(nipple_mask, kernel_size=2, stride=2)
        nipple_mask = F.max_pool2d(nipple_mask, kernel_size=2, stride=2)
        nipple_mask = F.max_pool2d(nipple_mask, kernel_size=2, stride=2)
        nipple_mask = nipple_mask * attention_mask
        nipple_x = nipple_mask * x
        nipple_analysis = self.nipple_analysis_cnn(nipple_x)
        
        _, nipple_coordinates = torch.max(nipple_analysis, dim=1)
        nipple_coordinates = (nipple_coordinates.float() / torch.tensor(np.prod(nipple_mask.size()[2:])).float()) * breast_measurement
        nipple_coordinates = nipple_coordinates.int() + torch.tensor([[grid_size[0] // 2], [grid_size[1] // 2]])

        return nipple_coordinates

    def body_analysis_cnn(self, x):
        layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        return layers(x)

    def nipple_analysis_cnn(self, x):
        layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )
        return layers(x)
    

class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
