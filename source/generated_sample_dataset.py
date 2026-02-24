import os
import glob
import cv2
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURAÇÕES DA IMAGEM E ESTILO IEEE
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 12,
    "figure.figsize": (14, 3.5), # Mais largo para acomodar 8 colunas confortavelmente
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05
})

# Pastas onde estão suas imagens
DIR_ALL = "./dataset/leukemia"  
DIR_HEM = "./dataset/healthy"  

# Tamanho exato que todas as imagens terão 
IMG_SIZE = (256, 256)

def create_dataset_grid():
    # Pegar as 8 primeiras imagens 
    all_images = glob.glob(os.path.join(DIR_ALL, "*.tif"))[:8] 
    hem_images = glob.glob(os.path.join(DIR_HEM, "*.tif"))[:8]
    
    if len(all_images) < 8 or len(hem_images) < 8:
        print(f"Erro: Precisamos de 8 imagens. Encontradas {len(all_images)} ALL e {len(hem_images)} Healthy.")
        return

    # Criar a grade 2 linhas x 8 colunas
    fig, axes = plt.subplots(2, 8)
    
    # --- Linha 0: Leucemia (ALL) ---
    for i in range(8):
        img = cv2.imread(all_images[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        
        # FORÇA o redimensionamento para garantir que todas tenham o mesmo tamanho
        img = cv2.resize(img, IMG_SIZE) 
        
        axes[0, i].imshow(img)
        axes[0, i].axis('off') 
        
        #if i == 0:
        #    axes[0, i].text(-0.2, 0.5, 'ALL\n(Pathogenic)', fontsize=12, fontweight='bold', 
        #                    va='center', ha='right', transform=axes[0, i].transAxes)

    # --- Linha 1: Saudáveis (Healthy) ---
    for i in range(8):
        img = cv2.imread(hem_images[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # FORÇA o redimensionamento
        img = cv2.resize(img, IMG_SIZE)
        
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        
        #if i == 0:
        #    axes[1, i].text(-0.2, 0.5, 'Healthy\n(Normal)', fontsize=12, fontweight='bold', 
        #                    va='center', ha='right', transform=axes[1, i].transAxes)

    # Ajustar espaçamento colado
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # Salvar nos formatos PDF (para o LaTeX) e PNG
    plt.savefig('fig_dataset_samples.pdf', format='pdf', bbox_inches='tight') 
    plt.savefig('fig_dataset_samples.png', format='png', bbox_inches='tight', dpi=300)
    
    print("Sucesso! Imagens geradas e redimensionadas para o mesmo tamanho.")
    plt.show()

if __name__ == "__main__":
    create_dataset_grid()
