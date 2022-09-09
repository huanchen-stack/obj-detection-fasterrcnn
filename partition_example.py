import torch
import torchvision

from util.partition_manager import PartitionManager, partition_manager


class Resnet(torch.nn.Module):

    def __init__(self):
        super(Resnet, self).__init__()

        self.model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
        
        ################################################################
        ################### ADD THIS SECTION TO CODE ###################
        ################################################################
        self.filtering = False
        # self.exec_labels = {'res_layer3', 'res_conv1', 'res_layer4', 'res_layer2', 'res_relu', 'res_fc', 'res_layer1', 'res_avgpool', 'res_bn1', 'res_maxpool'}
        self.exec_labels = {'res_layer3', 'res_layer1', 'res_avgpool'}
        self.args = {}
        ################################################################


    def simulate(self, x):
        self.args['res_x'] = x.clone().detach()
        with torch.no_grad():
            x = x.clone().detach()
            prevname = 'x'
            for name, layer in self.model.named_children():
                if name == 'fc':
                    # all args are loaded/stored from/into self.args 
                    self.args[f"res_{prevname}"] = torch.flatten(self.args[f"res_{prevname}"], 1)
                    # x = torch.flatten(x, 1)
                
                # The following segment executes
                #   x = layer(x)
                # 0. the input arg here is "res_prevname"
                # 1. partition_manager convert "res_prevname" to self.args["res_prevname"]
                # 2. when skimming, parition_manager store the intermediate results to self.args
                #       using key=res_suffix
                # 3. when filtering, partition_manager only executes the layers in self.exec_labels
                @partition_manager(self, filtering=self.filtering, suffix=f"_{name}")
                def res(x):
                    return layer(x)
                res(f"res_{prevname}")

                prevname = name

    def forward(self, x):
        self.simulate(x)
        self.filtering = True
        self.simulate(x)


if __name__ == '__main__':
    resnet = Resnet()
    dummy = torch.rand(1, 3, 224, 224)
    resnet(dummy)

