# KidneyGemini
腎糸球体画像にgeminiのVLMつかう


To perform K-nearest neighbors (KNN) retrieval, we embedded glomerular images using four different encoders:
ImageNet-pretrained ResNet50 (ImageNet-ResNet50: weights from https://huggingface.co/timm/resnet50.a1_in1k)
ImageNet-ViT-Base (ImageNet-ViT-Base: weights from https://huggingface.co/google/vit-base-patch16-224)
Vision Transformer-Base with self-supervised learning on PAS-stained glomerular images (DINO-glo-ViT-Base: weights from https://www.kaggle.com/datasets/abebe9849/sslglomerular-images-weights?select=checkpoint0600.pth)
Vision Transformer-Large trained on diverse pathological images (UNI-v1: weights from https://huggingface.co/MahmoodLab/UNI))
