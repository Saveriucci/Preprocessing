import os
import csv

# Percorso dataset
dataset_path = r"..\Progetto\Dataset\colon lung 15000"

# Percorso di salvataggio del CSV
csv_path = os.path.join(dataset_path, "dataset_labels.csv")

# Apriamo il file CSV in modalità scrittura
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_path", "organ", "label"])  # Intestazione del CSV

    # Scansione delle cartelle principali (colon, lung)
    for organ in os.listdir(dataset_path):
        organ_path = os.path.join(dataset_path, organ)

        if os.path.isdir(organ_path):  
            print(f"✅ Trovata cartella organo: {organ}")

            # Scansione delle sottocartelle (maligno, benigno, squamoso)
            for label in os.listdir(organ_path):
                label_path = os.path.join(organ_path, label)

                if os.path.isdir(label_path):
                    print(f"   📂 Trovata sottocartella etichetta: {label}")

                    # Scansione immagini
                    for image_file in os.listdir(label_path):
                        image_path = os.path.join(label_path, image_file)

                        # Controllo se è un file immagine valido
                        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                            writer.writerow([image_path, organ, label])  # Scrive direttamente nel CSV
                        else:
                            print(f"   ❌ File non valido (ignorato): {image_path}")

print(f"✅ CSV creato con successo! Salvato in: {csv_path}")
