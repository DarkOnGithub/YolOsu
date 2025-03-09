# import torch
# from model.net import YoloV3, config

# def test():
#     num_classes = 20
#     model = YoloV3(classes=num_classes,config=config)
#     img_size = 416
#     x = torch.randn((2, 3, img_size, img_size))
#     out = model(x)
#     assert out[0].shape == (2, 3, img_size//32, img_size//32, 5 + num_classes)
#     assert out[1].shape == (2, 3, img_size//16, img_size//16, 5 + num_classes)
#     assert out[2].shape == (2, 3, img_size//8, img_size//8, 5 + num_classes)

# test()

from parser.osz_parser import parse_osz_file
print(parse_osz_file("./maps/test.osz"))