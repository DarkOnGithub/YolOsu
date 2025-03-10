from torch import nn
import torch




config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8], 
    (1024, 3, 2),
    ["B", 4],
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
] 

class CNN(nn.Module):
    def __init__(self, in_features, out_features, batch_norm = False, **kwargs):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, bias=not batch_norm, **kwargs)
        self.bn = nn.BatchNorm2d(out_features) if batch_norm else None
        self.relu = nn.LeakyReLU(0.1)
        
    def forward(self, X):
        if self.bn:
            return self.relu(self.bn(self.conv(X)))
        return self.conv(X)
    
    
class Residual(nn.Module):
    def __init__(self, out_channels, is_residual = False, num_repeats = 1):
        super(Residual, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNN(out_channels, out_channels // 2, kernel_size = 1),
                    CNN(out_channels // 2, out_channels, kernel_size = 3, padding = 1)
                )
            ]
        self.use_residual = is_residual
        self.repeats = num_repeats
        
    def forward(self, X):
        for layer in self.layers:
            X = layer(X) + self.use_residual * X
        return X
    
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, classes):
        super(ScalePrediction, self).__init__()
        self.prediction = nn.Sequential(
            CNN(in_channels, in_channels * 2, kernel_size = 3, padding = 1),
            CNN(in_channels * 2, (classes + 5) * 3, batch_norm=False, kernel_size = 1)
        )
        self.classes = classes
    
    def forward(self, X):
        
        
        return (self.prediction(X)
                .reshape(X.shape[0], 3, self.classes + 5, X.shape[2],X.shape[3])
                .permute(0, 1, 3, 4, 2)
        )

class YoloV3(nn.Module):
    def __init__(self, in_channels = 3, classes = 0, config = config):
        super(YoloV3, self).__init__()
        self.classes = classes
        self.in_channels = in_channels
        self.layers = self.create_layers(config)
        
        
    def forward(self, X):
        out = []
        routes = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                out.append(layer(X))
                continue
            X = layer(X)
            if isinstance(layer, Residual) and layer.repeats == 8:
                routes.append(X)
            elif isinstance(layer, nn.Upsample):
                X = torch.cat([X, routes[-1]],  dim=1)
                routes.pop()
        return out
    
    def create_layers(self, config):
        in_channels = self.in_channels
        layers = nn.ModuleList()
        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(CNN(in_channels, out_features=out_channels, kernel_size=kernel_size, 
                                stride=stride, padding=(1 if kernel_size == 3 else 0)))                
                in_channels = out_channels
            elif isinstance(module, list):
                repeats = module[1]
                layers.append(Residual(in_channels, num_repeats=repeats))
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        Residual(in_channels, is_residual=False, num_repeats=1),
                        CNN(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, classes=self.classes)
                    ]
                    in_channels = in_channels // 2
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3
        return layers
                