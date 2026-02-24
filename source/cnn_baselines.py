import os
import glob
import csv
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
DATA_DIR = "./dataset"  
BATCH_SIZE = 16
EPOCHS = 30
K_FOLDS = 5
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. DATASET CUSTOMIZADO
# ==========================================
class ALLDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        from PIL import Image
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

def get_transforms():
    # Transformações idênticas ao pré-processamento clássico para ser justo
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

# ==========================================
# 3. CARREGAMENTO DE MODELOS (TRANSFER LEARNING)
# ==========================================
def build_model(model_name):
    # --- MODELOS CLÁSSICOS ---
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 1) # Binário
        
    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)
        
    elif model_name == "squeezenet":
        model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1,1), stride=(1,1))
        model.num_classes = 1

    # --- MODELOS MODERNOS ---
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, 1)

    # --- VISION TRANSFORMERS ---
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, 1)

    elif model_name == "swin_t":
        model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(model.head.in_features, 1)

    else:
        raise ValueError(f"Model {model_name} not supported")
        
    return model.to(DEVICE)

# ==========================================
# 4. TREINAMENTO E AVALIAÇÃO
# ==========================================
def train_and_evaluate(model_name, file_paths, labels, csv_filename="cnn_baselines_results.csv"):
    print(f"\n[{model_name.upper()}] Iniciando Stratified {K_FOLDS}-Fold CV...")
    
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    train_transform, val_transform = get_transforms()
    
    metrics = {'acc': [], 'mcc': [], 'auc': [], 'prauc': []}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels)):
        train_dataset = ALLDataset(np.array(file_paths)[train_idx], np.array(labels)[train_idx], transform=train_transform)
        val_dataset = ALLDataset(np.array(file_paths)[val_idx], np.array(labels)[val_idx], transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = build_model(model_name)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        best_auc = 0.0
        best_preds, best_probs, best_true = [], [], []
        
        for epoch in range(EPOCHS):
            model.train()
            for images, targets in train_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images).squeeze(-1)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_probs, val_targets = [], []
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(DEVICE)
                    outputs = model(images).squeeze(-1)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    val_probs.extend(probs)
                    val_targets.extend(targets.numpy())
            
            val_preds = (np.array(val_probs) >= 0.5).astype(int)
            
            # Tratamento de segurança para o caso de o fold ter gerado predições estranhas
            try:
                current_auc = roc_auc_score(val_targets, val_probs)
            except ValueError:
                current_auc = 0.0

            if current_auc > best_auc:
                best_auc = current_auc
                best_preds = val_preds
                best_probs = val_probs
                best_true = val_targets
                
        # Se não convergiu nada, usa a última época
        if len(best_true) == 0:
            best_preds, best_probs, best_true = val_preds, val_probs, val_targets

        acc = accuracy_score(best_true, best_preds)
        mcc = matthews_corrcoef(best_true, best_preds)
        prauc = average_precision_score(best_true, best_probs)
        
        metrics['acc'].append(acc)
        metrics['mcc'].append(mcc)
        metrics['auc'].append(best_auc)
        metrics['prauc'].append(prauc)
        
        print(f"Fold {fold+1} | Acc: {acc:.4f} | MCC: {mcc:.4f} | AUC: {best_auc:.4f}")

    mean_acc = np.mean(metrics['acc'])
    mean_mcc = np.mean(metrics['mcc'])
    mean_auc = np.mean(metrics['auc'])
    mean_prauc = np.mean(metrics['prauc'])

    print(f"\n--- Resultados Finais: {model_name.upper()} ---")
    print(f"Accuracy: {mean_acc:.4f} +/- {np.std(metrics['acc']):.4f}")
    print(f"MCC:      {mean_mcc:.4f} +/- {np.std(metrics['mcc']):.4f}")
    print(f"AUC-ROC:  {mean_auc:.4f} +/- {np.std(metrics['auc']):.4f}")
    print(f"PR-AUC:   {mean_prauc:.4f} +/- {np.std(metrics['prauc']):.4f}")
    print("-" * 40)

    # SALVAR NO CSV
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'model', 'fold', 'accuracy', 'mcc', 'auc_roc', 'pr_auc'])
        
        for i in range(K_FOLDS):
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_name,
                f"Fold_{i+1}",
                round(metrics['acc'][i], 4),
                round(metrics['mcc'][i], 4),
                round(metrics['auc'][i], 4),
                round(metrics['prauc'][i], 4)
            ])
            
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            "MEAN_CV",
            round(mean_acc, 4),
            round(mean_mcc, 4),
            round(mean_auc, 4),
            round(mean_prauc, 4)
        ])

# ==========================================
# 5. EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == '__main__':
    DATA_DIR = "./dataset"  
    
    all_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif"]:
        all_files.extend(glob.glob(os.path.join(DATA_DIR, "**", ext), recursive=True))
        
    if len(all_files) == 0:
        print("Erro: Nenhuma imagem encontrada. Verifique o DATA_DIR.")
    else:
        labels = []
        for f in all_files:
            nome_arquivo = os.path.basename(f).lower()
            pasta_pai = os.path.basename(os.path.dirname(f)).lower()
            
            if "_1." in nome_arquivo:
                labels.append(1)
            elif "_0." in nome_arquivo:
                labels.append(0)
            elif "all" in pasta_pai or "leukemia" in pasta_pai or "blast" in pasta_pai:
                labels.append(1)
            elif "normal" in pasta_pai or "hem" in pasta_pai:
                labels.append(0)
            else:
                labels.append(1 if "all" in nome_arquivo else 0)

        total = len(labels)
        positivos = sum(labels)
        negativos = total - positivos
        
        print(f"Total de Imagens Encontradas: {total}")
        print(f" - Classe 1 (ALL / Leucemia): {positivos}")
        print(f" - Classe 0 (Saudáveis):      {negativos}")
        
        if positivos == 0 or negativos == 0:
            print("ERRO CRÍTICO: O script não conseguiu encontrar imagens de ambas as classes.")
        else:
            print("\nIniciando testes com Deep Learning...")
            
            # ATUALIZADO: Testando desde as clássicas até Swin Transformer
            models_to_test = ["resnet18", "vgg16", "efficientnet_b0", "densenet121", "swin_t"]
            
            csv_file = "cnn_baselines_modern_results.csv"
            if os.path.exists(csv_file):
                os.remove(csv_file)
                
            for m in models_to_test:
                train_and_evaluate(m, all_files, labels, csv_filename=csv_file)
            
            print(f"\nTodos os resultados foram salvos com sucesso em: {csv_file}")
