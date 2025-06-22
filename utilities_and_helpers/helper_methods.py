import timm
from transformers import AutoModel
from models import TwoLayerNN, ThreeLayerNN
import torch

#instantiates a model based on the modelname with the given parameters
def inst_model(modelname, input_size, hidden_size, hidden_size1, hidden_size2, device):
    flatten_patches = False
    if modelname == "TwoLayerNN":
        model = TwoLayerNN(input_size, hidden_size, 2)
        flatten_patches = True
    elif modelname == "ThreeLayerNN":
        model = ThreeLayerNN(input_size, hidden_size1, hidden_size2, 2)
        flatten_patches = True
    elif modelname == "ResNet18":
        model = timm.create_model('resnet18', pretrained=True, num_classes=2)
    elif modelname == "ResNet50":
        model = timm.create_model('resnet50', pretrained=True, num_classes=2)
    elif modelname == "ResNet101":
        model = timm.create_model('resnet101', pretrained=True, num_classes=2)
    elif modelname == "DenseNet121":
        model = timm.create_model('densenet121', pretrained=True, num_classes=2)
    elif modelname == "Dino":
        # Create the DINO model
        model = timm.create_model(
            'vit_base_patch16_224.dino',
            pretrained=True,
            num_classes=2,  # Remove classifier (no classification head)
        )
        # Freeze all layers except the classification head
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True

    elif modelname == 'Dinov2':
        # Load the base model       
        model = AutoModel.from_pretrained('facebook/dinov2-base')

        
    else:
        print("model not found")
        return
    return model.to(device), flatten_patches



if __name__ == '__main__':
    #this script can alternatively be run to instantiate and save out of the box models for later use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    basepath = "specify savepath here"  # Replace with your desired save path

    # Instantiate ResNet18
    resnet18_model, _ = inst_model("ResNet18", None, None, None, None, device)
    resnet18_path = f"{basepath}/OOTB_ResNet18_epoch_0_lr_0.pth"
    torch.save(resnet18_model.state_dict(), resnet18_path)

    # Instantiate Dino
    dino_model, _ = inst_model("Dino", None, None, None, None, device)
    dino_path = f"{basepath}/OOTB_Dino_epoch_0_lr_0.pth"
    torch.save(dino_model.state_dict(), dino_path)

    # Instantiate Dinov2
    dinov2_model, _ = inst_model("Dinov2", None, None, None, None, device)
    dinov2_path = f"{basepath}/OOTB_Dinov2_epoch_0_lr_0.pth"
    torch.save(dinov2_model.state_dict(), dinov2_path)

    print(f"Models saved to:\nResNet18: {resnet18_path}\nDino: {dino_path}\nDinov2: {dinov2_path}")
