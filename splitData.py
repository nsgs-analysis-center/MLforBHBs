import argparse
import math
import random
import shutil
import uuid
from pathlib import Path

def resplitDataset(baseDir, splitRatio):
    basePath = Path(baseDir)
    if not basePath.exists():
        print(f"Error: Directory '{baseDir}' does not exist.")
        return

    print(f"Scanning '{baseDir}' for valid image/label/param triplets...")

    # 1. Define internal paths
    imageDir = basePath / 'images'
    labelDir = basePath / 'labels'
    paramDir = basePath / 'params'

    # recursively gather all images regardless of which split folder they are currently in
    allImages = list(imageDir.rglob("*.jpg")) + list(imageDir.rglob("*.png"))

    validTriplets = []

    # 2. Filter for complete data
    for imgPath in allImages:
        splitFolder = imgPath.parent.name  # 'train', 'val', or 'test'
        stem = imgPath.stem

        lblPath = labelDir / splitFolder / f"{stem}.txt"
        yamlPath = paramDir / splitFolder / f"{stem}.yaml"

        # only accept images that have their matching YOLO label and parameter file
        if lblPath.exists() and yamlPath.exists():
            validTriplets.append((imgPath, lblPath, yamlPath))
        else:
            print(f"Warning: Incomplete data for {stem}. Skipping to prevent training crashes.")

    totalValid = len(validTriplets)
    if totalValid == 0:
        print("No valid data found to split.")
        return

    print(f"Found {totalValid} complete dataset items. Shuffling...")

    # 3. Randomize the dataset
    random.shuffle(validTriplets)

    # 4. Calculate exact split counts (handling edge cases where percentages don't perfectly equal 100)
    trainPct, valPct, testPct = splitRatio
    normFactor = (trainPct + valPct + testPct) / 100.0
    trainPct /= normFactor
    valPct /= normFactor

    numTrain = math.floor(totalValid * trainPct / 100)
    numVal = math.floor(totalValid * valPct / 100)
    numTest = totalValid - numTrain - numVal  # guarantees no remainder is left behind

    print(f"New split counts -> Train: {numTrain}, Val: {numVal}, Test: {numTest}")

    splitsConfig = [
        ('train', numTrain),
        ('val', numVal),
        ('test', numTest)
    ]

    # 5. Safe Two-Pass Rename (Staging Phase)
    print("Moving files to a safe staging area to prevent overwrite collisions...")
    stagingDir = basePath / 'staging'
    if stagingDir.exists():
        shutil.rmtree(stagingDir)
    stagingDir.mkdir()

    stagedTriplets = []
    for imgPath, lblPath, yamlPath in validTriplets:
        # use random gibberish to guarantee absolute safety
        tempId = str(uuid.uuid4())
        
        newImg = stagingDir / f"{tempId}{imgPath.suffix}"
        newLbl = stagingDir / f"{tempId}.txt"
        newYaml = stagingDir / f"{tempId}.yaml"

        imgPath.rename(newImg)
        lblPath.rename(newLbl)
        yamlPath.rename(newYaml)

        stagedTriplets.append((newImg, newLbl, newYaml))

    # 6. Final Placement Phase
    print("Distributing into train/val/test folders...")
    currentIdx = 0

    for splitName, count in splitsConfig:
        # ensure target directories exist
        (imageDir / splitName).mkdir(parents=True, exist_ok=True)
        (labelDir / splitName).mkdir(parents=True, exist_ok=True)
        (paramDir / splitName).mkdir(parents=True, exist_ok=True)

        for i in range(count):
            stgImg, stgLbl, stgYaml = stagedTriplets[currentIdx]

            # create clean, sequential names with no gaps
            finalName = f"{splitName}_img_{i:04d}"

            stgImg.rename(imageDir / splitName / f"{finalName}{stgImg.suffix}")
            stgLbl.rename(labelDir / splitName / f"{finalName}.txt")
            stgYaml.rename(paramDir / splitName / f"{finalName}.yaml")

            currentIdx += 1

    # 7. Cleanup
    shutil.rmtree(stagingDir)
    print("Dataset successfully shuffled and re-split!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseDir', type = str, required = True, help = 'Path to the dataset directory (e.g., imageData5).')
    parser.add_argument('--trainValTest', type = str, default = '[70, 20, 10]', help = 'Split percentages in a python list format.')

    args = parser.parse_args()

    splitList = [float(x) for x in args.trainValTest.strip('[]').replace(',', '').split()]
    
    resplitDataset(baseDir = args.baseDir, splitRatio = splitList)