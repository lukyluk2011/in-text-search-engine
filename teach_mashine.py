import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

class TextDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx].todense(), dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def load_data_from_directory(directory):
    data = []
    labels = []
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            text_file_path = os.path.join(subdir_path, 'text.txt')
            if os.path.isfile(text_file_path):
                with open(text_file_path, 'r', encoding='utf-8') as f:
                    main_text = f.read()
                
                for label in ['je', 'není']:
                    label_dir = os.path.join(subdir_path, label)
                    if os.path.isdir(label_dir):
                        for txt_file in os.listdir(label_dir):
                            txt_file_path = os.path.join(label_dir, txt_file)
                            if os.path.isfile(txt_file_path) and txt_file_path.endswith('.txt'):
                                with open(txt_file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    data.append(content)
                                    labels.append(1 if label == 'je' else 0)
    return data, labels

def vectorize_data(data):
    vectorizer = CountVectorizer()
    vectorized_data = vectorizer.fit_transform(data)
    return vectorized_data, vectorizer

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Načtení a příprava dat
train_data_dir = './train_data'
data, labels = load_data_from_directory(train_data_dir)

# Rozdělení dat na trénovací a testovací sadu
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Vektorizace textů
vectorized_train_data, vectorizer = vectorize_data(train_data)
vectorized_test_data = vectorizer.transform(test_data)

# Příprava datasetů a dataloaderů
train_dataset = TextDataset(vectorized_train_data, train_labels)
test_dataset = TextDataset(vectorized_test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Výpis velikostí datasetů
print(f'Trénovací sada obsahuje {len(train_dataset)} vzorků')
print(f'Testovací sada obsahuje {len(test_dataset)} vzorků')

# Definice modelu, loss funkce a optimizeru
input_dim = vectorized_train_data.shape[1]
model = SimpleNN(input_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trénování modelu
for epoch in range(5):  # Tréninkový cyklus přes 5 epoch
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

print('Trénink modelu byl úspěšně dokončen.')

# Testování modelu
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Úspěšnost sítě na testovacích datech: %d %%' % (100 * correct / total))

# Uložení trénovaného modelu ve formátu ONNX
dummy_input = torch.randn(1, input_dim)  # Vstupní tensor pro export
onnx_path = "./text_recognition_model.onnx"
torch.onnx.export(model, dummy_input, onnx_path, verbose=True)

print('Model byl uložen ve formátu ONNX do souboru:', onnx_path)
