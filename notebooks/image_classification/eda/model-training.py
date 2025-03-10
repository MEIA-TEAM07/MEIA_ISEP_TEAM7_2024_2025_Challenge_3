import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor
import cv2
import TurboJPEG

# Etapa 1: Carregamento do TurboJPEG ou OpenCV
with tqdm(total=1, desc="Carregando TurboJPEG/OpenCV", bar_format="{l_bar}{bar} [{elapsed}]") as pbar:
    try:
        
        dll_path = r"C:\libjpeg-turbo64\bin\turbojpeg.dll"  # Ajuste conforme necessário
        if os.path.exists(dll_path):
            jpeg = TurboJPEG(lib_path=dll_path)
            print("\nTurboJPEG carregado com sucesso!")
        else:
            raise FileNotFoundError(f"DLL não encontrada em: {dll_path}")
    except (ImportError, FileNotFoundError, RuntimeError) as e:
        print(f"\nErro ao carregar TurboJPEG: {e}. Usando OpenCV como fallback.")
        jpeg = None
    pbar.update(1)

# Carregar CSV
df = pd.read_csv('train/_annotations.csv')
image_folder = "train/"
X, y = [], []

def extract_color_histogram_features(image):
    """Extrai histogramas de cor no espaço HSV com 8 bins por canal."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    bins = 8
    hist_h = cv2.calcHist([h], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([s], [0], None, [bins], [0, 256])
    hist_v = cv2.calcHist([v], [0], None, [bins], [0, 256])
    hist_h = hist_h / hist_h.sum() if hist_h.sum() != 0 else np.zeros_like(hist_h)
    hist_s = hist_s / hist_s.sum() if hist_s.sum() != 0 else np.zeros_like(hist_s)
    hist_v = hist_v / hist_v.sum() if hist_v.sum() != 0 else np.zeros_like(hist_v)
    return np.hstack([hist_h.ravel(), hist_s.ravel(), hist_v.ravel()])

def process_image(row):
    img_path = os.path.join(image_folder, row['filename'])
    if not os.path.exists(img_path):
        print(f"\nImagem não encontrada: {img_path}")
        return None, None

    if jpeg is not None:
        try:
            with open(img_path, 'rb') as file:
                img = jpeg.decode(file.read())
        except Exception as e:
            print(f"\nErro ao carregar com TurboJPEG: {img_path}, erro: {e}")
            return None, None
    else:
        img = cv2.imread(img_path)

    if img is None:
        print(f"\nErro ao carregar a imagem: {img_path}")
        return None, None

    x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    h, w, _ = img.shape
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)
    roi = img[y_min:y_max, x_min:x_max]

    if roi.shape[0] < 1 or roi.shape[1] < 1:
        print(f"\nROI muito pequeno para processamento: {img_path}")
        return None, None

    roi_resized = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)
    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.equalizeHist(roi_gray)

    fd = hog(roi_gray, orientations=12, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    color_features = extract_color_histogram_features(roi_resized)
    combined_features = np.hstack([fd, color_features])
    return combined_features, row['class']

# Etapa 2: Processamento das Imagens
with ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_image, [row for _, row in df.iterrows()]), total=len(df), desc="Processando Imagens", bar_format="{l_bar}{bar} [{elapsed}]"))

# Filtrar resultados válidos
for fd, label in results:
    if fd is not None and label is not None:
        X.append(fd)
        y.append(label)

# Etapa 3: Conversão para Arrays NumPy
with tqdm(total=1, desc="Convertendo para Arrays NumPy", bar_format="{l_bar}{bar} [{elapsed}]") as pbar:
    X = np.array(X)
    y = np.array(y)
    pbar.update(1)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Etapa 4: Treinamento do Modelo com GridSearchCV
print("\nIniciando treinamento do modelo...")
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
param_grid = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)
with tqdm(total=1, desc="Treinando Modelo", bar_format="{l_bar}{bar} [{elapsed}]") as pbar:
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    pbar.update(1)

print("\nTreinamento concluído!")
print(f"Melhores parâmetros: {grid_search.best_params_}")

# Etapa 5: Avaliação do Modelo
with tqdm(total=1, desc="Avaliando Modelo", bar_format="{l_bar}{bar} [{elapsed}]") as pbar:
    y_pred = best_rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pbar.update(1)

print(f"\nAcurácia do modelo: {acc:.4f}")