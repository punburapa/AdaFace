import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import os
from PIL import Image
import pandas as pd
import net
import time
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 1. Load Model
# ==============================
def load_model(checkpoint_path):
    model = net.build_model('ir_101')
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cuda' if torch.cuda.is_available() else 'cpu'
    )
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_state_dict[key[6:]] = value
        elif key.startswith('head.'):
            pass
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
        print("✅ Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("⚠️ Using CPU")
    print("✅ Model Loaded!")
    return model

# ==============================
# 2. Preprocess
# ==============================
def preprocess(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((112, 112))
        img = np.array(img, dtype=np.float32)
        img = (img / 255.0 - 0.5) / 0.5
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    except Exception as e:
        print(f"❌ Error loading {img_path}: {e}")
        return None

# ==============================
# 3. Read Pairs
# ==============================
def read_pairs(lfw_root, match_file, mismatch_file):
    pairs = []
    same  = 0
    diff  = 0

    # Same person
    df_match = pd.read_csv(match_file)
    for _, row in df_match.iterrows():
        try:
            name = str(row['name']).strip()
            idx1 = int(row['imagenum1'])
            idx2 = int(row['imagenum2'])
            img1 = os.path.join(lfw_root, name, f"{name}_{idx1:04d}.jpg")
            img2 = os.path.join(lfw_root, name, f"{name}_{idx2:04d}.jpg")
            pairs.append((img1, img2, 1))
            same += 1
        except:
            continue

    # Different person
    df_mismatch = pd.read_csv(mismatch_file)
    for _, row in df_mismatch.iterrows():
        try:
            name1 = str(row['name']).strip()
            idx1  = int(row['imagenum1'])
            name2 = str(row['name.1']).strip()
            idx2  = int(row['imagenum2'])
            img1  = os.path.join(lfw_root, name1, f"{name1}_{idx1:04d}.jpg")
            img2  = os.path.join(lfw_root, name2, f"{name2}_{idx2:04d}.jpg")
            pairs.append((img1, img2, 0))
            diff += 1
        except:
            continue

    print(f"  ✅ Same person pairs     : {same}")
    print(f"  ✅ Different person pairs: {diff}")
    return pairs

# ==============================
# 4. Evaluate
# ==============================
def evaluate(model, pairs):
    labels                = []
    similarities          = []
    skipped               = 0
    total_preprocess_time = 0
    total_infer_time      = 0
    total_pairs_evaluated = 0

    print(f"📊 Total pairs: {len(pairs)}")
    t_total_start = time.perf_counter()

    for i, (img1_path, img2_path, label) in enumerate(pairs):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(pairs)}...")

        t_pre = time.perf_counter()
        img1  = preprocess(img1_path)
        img2  = preprocess(img2_path)
        total_preprocess_time += time.perf_counter() - t_pre

        if img1 is None or img2 is None:
            skipped += 1
            continue

        if torch.cuda.is_available():
            img1 = img1.cuda()
            img2 = img2.cuda()

        t_inf = time.perf_counter()
        with torch.no_grad():
            feat1, _ = model(img1)
            feat2, _ = model(img2)
        total_infer_time += time.perf_counter() - t_inf

        feat1 = feat1.cpu().numpy()
        feat2 = feat2.cpu().numpy()

        sim = np.dot(feat1, feat2.T) / (
            np.linalg.norm(feat1) * np.linalg.norm(feat2)
        )
        similarities.append(float(sim))
        labels.append(label)
        total_pairs_evaluated += 1

    total_time = time.perf_counter() - t_total_start
    print(f"⚠️  Skipped: {skipped}")

    if len(similarities) == 0:
        print("❌ ไม่มี pairs!")
        return None

    similarities_arr = np.array(similarities)
    labels_arr       = np.array(labels)

    best_acc    = 0
    best_thresh = 0
    for thresh in np.arange(0.0, 1.0, 0.01):
        preds = (similarities_arr > thresh).astype(int)
        acc   = accuracy_score(labels_arr, preds)
        if acc > best_acc:
            best_acc    = acc
            best_thresh = thresh

    timing = {
        'total_time'       : total_time,
        'preprocess_time'  : total_preprocess_time,
        'inference_time'   : total_infer_time,
        'pairs_evaluated'  : total_pairs_evaluated,
        'avg_preprocess_ms': (total_preprocess_time / (total_pairs_evaluated * 2)) * 1000,
        'avg_inference_ms' : (total_infer_time      / (total_pairs_evaluated * 2)) * 1000,
        'throughput'       : total_pairs_evaluated / total_time
    }

    return {
        'accuracy'    : best_acc,
        'threshold'   : best_thresh,
        'similarities': similarities_arr,
        'labels'      : labels_arr,
        'timing'      : timing
    }

# ==============================
# 5. Biometric Metrics
# ==============================
def compute_biometric_metrics(similarities, labels):
    genuine_scores  = similarities[labels == 1]
    impostor_scores = similarities[labels == 0]

    print(f"  Genuine  pairs : {len(genuine_scores)}")
    print(f"  Impostor pairs : {len(impostor_scores)}")

    thresholds = np.arange(0.0, 1.0, 0.001)
    FARs = []
    FRRs = []
    for thresh in thresholds:
        FAR = np.sum(impostor_scores >= thresh) / len(impostor_scores)
        FRR = np.sum(genuine_scores  <  thresh) / len(genuine_scores)
        FARs.append(FAR)
        FRRs.append(FRR)
    FARs = np.array(FARs)
    FRRs = np.array(FRRs)

    # EER
    try:
        eer_threshold = brentq(
            lambda x: interp1d(thresholds, FARs)(x) -
                      interp1d(thresholds, FRRs)(x),
            thresholds, thresholds[-1]
        )
        EER = float(interp1d(thresholds, FARs)(eer_threshold))
    except:
        idx           = np.argmin(np.abs(FARs - FRRs))
        EER           = float((FARs[idx] + FRRs[idx]) / 2)
        eer_threshold = float(thresholds[idx])

    # ROC + AUC
    fpr, tpr, _ = roc_curve(labels, similarities)
    AUC         = auc(fpr, tpr)

    try:
        TAR_at_FAR01 = float(interp1d(fpr, tpr)(0.001))
    except:
        TAR_at_FAR01 = 0.0
    try:
        TAR_at_FAR1  = float(interp1d(fpr, tpr)(0.01))
    except:
        TAR_at_FAR1  = 0.0

    idx        = np.argmin(np.abs(thresholds - eer_threshold))
    FAR_at_EER = float(FARs[idx])
    FRR_at_EER = float(FRRs[idx])

    return {
        'EER'          : EER,
        'EER_threshold': float(eer_threshold),
        'AUC'          : float(AUC),
        'TAR_at_FAR01' : TAR_at_FAR01,
        'TAR_at_FAR1'  : TAR_at_FAR1,
        'FAR_at_EER'   : FAR_at_EER,
        'FRR_at_EER'   : FRR_at_EER,
    }

# ==============================
# 6. Main
# ==============================
if __name__ == "__main__":
    CHECKPOINT    = "pretrained/adaface_ir101_webface4m.ckpt"
    LFW_ROOT      = "data/lfw/lfw-deepfunneled"
    MATCH_FILE    = "data/matchpairsDevTest.csv"
    MISMATCH_FILE = "data/mismatchpairsDevTest.csv"

    assert os.path.exists(CHECKPOINT),    f"❌ ไม่เจอ: {CHECKPOINT}"
    assert os.path.exists(LFW_ROOT),      f"❌ ไม่เจอ: {LFW_ROOT}"
    assert os.path.exists(MATCH_FILE),    f"❌ ไม่เจอ: {MATCH_FILE}"
    assert os.path.exists(MISMATCH_FILE), f"❌ ไม่เจอ: {MISMATCH_FILE}"

    print("⏱️ Loading model...")
    t_load_start = time.perf_counter()
    model        = load_model(CHECKPOINT)
    t_load_end   = time.perf_counter()

    print("\n📂 Reading pairs...")
    pairs = read_pairs(LFW_ROOT, MATCH_FILE, MISMATCH_FILE)
    print(f"✅ Found {len(pairs)} pairs")

    print("\n🔄 Evaluating...")
    results = evaluate(model, pairs)

    print("\n📊 Computing Biometric Metrics...")
    metrics = compute_biometric_metrics(
        results['similarities'],
        results['labels']
    )

    t = results['timing']
    print("\n" + "="*55)
    print("    AdaFace IR-101 (WebFace4M) | LFW Results")
    print("="*55)
    print(f"  {'Metric':<25} {'Value':>15}")
    print("-"*55)
    print(f"  {'Accuracy':<25} {results['accuracy']*100:>14.2f}%")
    print(f"  {'Best Threshold':<25} {results['threshold']:>15.3f}")
    print(f"  {'EER':<25} {metrics['EER']*100:>14.2f}%")
    print(f"  {'EER Threshold':<25} {metrics['EER_threshold']:>15.3f}")
    print(f"  {'AUC':<25} {metrics['AUC']:>15.4f}")
    print(f"  {'TAR @ FAR=0.1%':<25} {metrics['TAR_at_FAR01']*100:>14.2f}%")
    print(f"  {'TAR @ FAR=1.0%':<25} {metrics['TAR_at_FAR1']*100:>14.2f}%")
    print(f"  {'FAR @ EER':<25} {metrics['FAR_at_EER']*100:>14.2f}%")
    print(f"  {'FRR @ EER':<25} {metrics['FRR_at_EER']*100:>14.2f}%")
    print("-"*55)
    print(f"  {'Model Load Time':<25} {t_load_end-t_load_start:>13.3f} s")
    print(f"  {'Total Eval Time':<25} {t['total_time']:>13.3f} s")
    print(f"  {'Avg Preprocess':<25} {t['avg_preprocess_ms']:>12.2f} ms")
    print(f"  {'Avg Inference':<25} {t['avg_inference_ms']:>12.2f} ms")
    print(f"  {'Throughput':<25} {t['throughput']:>10.2f} pairs/s")
    print("="*55)