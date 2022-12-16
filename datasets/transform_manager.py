import torchvision.transforms as transforms

def get_transform(is_training=None,transform_type=None,pre=None):

    if is_training and pre:
        raise Exception('is_training and pre cannot be specified as True at the same time')

    if transform_type and pre:
        raise Exception('transform_type and pre cannot be specified as True at the same time')

    mean=[0.5,0.5,0.5]
    std=[0.5,0.5,0.5]

    normalize = transforms.Compose([
                                    transforms.Resize((128, 128)),  #128,216
                                    transforms.ToTensor(),
                                    ])

    if is_training:

        if transform_type == 0:
            size_transform = transforms.RandomResizedCrop(84)
        elif transform_type == 1:
            size_transform = transforms.RandomCrop(84,padding=8)
        else:
            # raise Exception('transform_type must be specified during training!')
            pass
        
        train_transform = transforms.Compose([normalize])
        return train_transform
    
    elif pre:
        return normalize
    
    else:
        
        if transform_type == 0:
            size_transform = transforms.Compose([transforms.Resize(92),
                                                transforms.CenterCrop(84)])
        elif transform_type == 1:
            size_transform = transforms.Compose([transforms.Resize([92,92]),
                                                transforms.CenterCrop(84)])
        elif transform_type == 2:
            # for tiered-imagenet and (tiered) meta-inat where val/test images are already 84x84
            return normalize

        else:
            # raise Exception('transform_type must be specified during inference if not using pre!')
            pass
        
        eval_transform = transforms.Compose([normalize])
        return eval_transform
