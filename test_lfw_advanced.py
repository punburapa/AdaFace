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
    return model

# ==============================
# 2. Preprocess
# ==============================
def preprocess_single(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((112, 112))
        img = np.array(img, dtype=np.float32)
        img = (img / 255.0 - 0.5) / 0.5
        return torch.from_numpy(img).permute(2, 0, 1)
    except:
        return None

# ==============================
# 3. Read Pairs
# ==============================
def read_pairs(lfw_root, match_file, mismatch_file):
    pairs = []
    same  = 0
    diff  = 0

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
# 4. Evaluate Baseline
# (FP32, Single Image)
# ==============================
def evaluate_baseline(model, pairs):
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
        img1  = preprocess_single(img1_path)
        img2  = preprocess_single(img2_path)
        total_preprocess_time += time.perf_counter() - t_pre

        if img1 is None or img2 is None:
            skipped += 1
            continue

        img1 = img1.unsqueeze(0).cuda()
        img2 = img2.unsqueeze(0).cuda()

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

    return _compute_results(
        np.array(similarities), np.array(labels),
        total_time, total_preprocess_time,
        total_infer_time, total_pairs_evaluated
    )

# ==============================
# 5. Evaluate Advanced
# (FP32 + Batch) ← เปลี่ยนจาก FP16
# ==============================
def evaluate_batch(model, pairs, batch_size=32):
    labels                = []
    similarities          = []
    skipped               = 0
    total_preprocess_time = 0
    total_infer_time      = 0
    total_pairs_evaluated = 0

    print(f"📊 Total pairs: {len(pairs)} | Batch size: {batch_size}")
    t_total_start = time.perf_counter()

    for batch_start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[batch_start: batch_start + batch_size]

        if batch_start % 500 == 0:
            print(f"  Processing {batch_start}/{len(pairs)}...")

        t_pre        = time.perf_counter()
        batch1       = []
        batch2       = []
        batch_labels = []

        for (img1_path, img2_path, label) in batch_pairs:
            t1 = preprocess_single(img1_path)
            t2 = preprocess_single(img2_path)
            if t1 is not None and t2 is not None:
                batch1.append(t1)
                batch2.append(t2)
                batch_labels.append(label)
            else:
                skipped += 1

        total_preprocess_time += time.perf_counter() - t_pre

        if len(batch1) == 0:
            continue

        # ✅ Stack เป็น batch
        batch1_tensor = torch.stack(batch1).cuda()
        batch2_tensor = torch.stack(batch2).cuda()

        # ✅ Inference ทั้ง batch ครั้งเดียว
        t_inf = time.perf_counter()
        with torch.no_grad():
            feats1, _ = model(batch1_tensor)
            feats2, _ = model(batch2_tensor)
        total_infer_time += time.perf_counter() - t_inf

        feats1 = feats1.cpu().numpy()
        feats2 = feats2.cpu().numpy()

        for j in range(len(batch_labels)):
            feat1 = feats1[j]
            feat2 = feats2[j]
            norm1 = np.linalg.norm(feat1)
            norm2 = np.linalg.norm(feat2)
            if norm1 == 0 or norm2 == 0:
                skipped += 1
                continue
            sim = np.dot(feat1, feat2) / (norm1 * norm2)
            similarities.append(float(sim))
            labels.append(batch_labels[j])
            total_pairs_evaluated += 1

    total_time = time.perf_counter() - t_total_start
    print(f"⚠️  Skipped: {skipped}")

    return _compute_results(
        np.array(similarities), np.array(labels),
        total_time, total_preprocess_time,
        total_infer_time, total_pairs_evaluated
    )

# ==============================
# 6. Compute Results
# ==============================
def _compute_results(similarities, labels,
                     total_time, preprocess_time,
                     infer_time, n_pairs):
    if len(similarities) == 0:
        return None

    best_acc    = 0
    best_thresh = 0
    for thresh in np.arange(0.0, 1.0, 0.01):
        preds = (similarities > thresh).astype(int)
        acc   = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc    = acc
            best_thresh = thresh

    timing = {
        'total_time'       : total_time,
        'preprocess_time'  : preprocess_time,
        'inference_time'   : infer_time,
        'pairs_evaluated'  : n_pairs,
        'avg_preprocess_ms': (preprocess_time / (n_pairs * 2)) * 1000,
        'avg_inference_ms' : (infer_time      / (n_pairs * 2)) * 1000,
        'throughput'       : n_pairs / total_time
    }

    return {
        'accuracy'    : best_acc,
        'threshold'   : best_thresh,
        'similarities': similarities,
        'labels'      : labels,
        'timing'      : timing
    }

# ==============================
# 7. Biometric Metrics
# ==============================
def compute_biometric_metrics(similarities, labels):
    genuine_scores  = similarities[labels == 1]
    impostor_scores = similarities[labels == 0]

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
# 8. Print Results
# ==============================
def print_results(name, results, metrics):
    t = results['timing']
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  {'Metric':<25} {'Value':>15}")
    print(f"{'-'*55}")
    print(f"  {'Accuracy':<25} {results['accuracy']*100:>14.2f}%")
    print(f"  {'Best Threshold':<25} {results['threshold']:>15.3f}")
    print(f"  {'EER':<25} {metrics['EER']*100:>14.2f}%")
    print(f"  {'EER Threshold':<25} {metrics['EER_threshold']:>15.3f}")
    print(f"  {'AUC':<25} {metrics['AUC']:>15.4f}")
    print(f"  {'TAR @ FAR=0.1%':<25} {metrics['TAR_at_FAR01']*100:>14.2f}%")
    print(f"  {'TAR @ FAR=1.0%':<25} {metrics['TAR_at_FAR1']*100:>14.2f}%")
    print(f"  {'FAR @ EER':<25} {metrics['FAR_at_EER']*100:>14.2f}%")
    print(f"  {'FRR @ EER':<25} {metrics['FRR_at_EER']*100:>14.2f}%")
    print(f"{'-'*55}")
    print(f"  {'Total Eval Time':<25} {t['total_time']:>13.3f} s")
    print(f"  {'Avg Preprocess':<25} {t['avg_preprocess_ms']:>12.2f} ms")
    print(f"  {'Avg Inference':<25} {t['avg_inference_ms']:>12.2f} ms")
    print(f"  {'Throughput':<25} {t['throughput']:>10.2f} pairs/s")
    print(f"{'='*55}")

# ==============================
# 9. Main
# ==============================
if __name__ == "__main__":
    CHECKPOINT    = "pretrained/adaface_ir101_webface4m.ckpt"
    LFW_ROOT      = "data/lfw/lfw-deepfunneled"
    MATCH_FILE    = "data/matchpairsDevTest.csv"
    MISMATCH_FILE = "data/mismatchpairsDevTest.csv"
    BATCH_SIZE    = 32

    assert os.path.exists(CHECKPOINT),    f"❌ ไม่เจอ: {CHECKPOINT}"
    assert os.path.exists(LFW_ROOT),      f"❌ ไม่เจอ: {LFW_ROOT}"
    assert os.path.exists(MATCH_FILE),    f"❌ ไม่เจอ: {MATCH_FILE}"
    assert os.path.exists(MISMATCH_FILE), f"❌ ไม่เจอ: {MISMATCH_FILE}"

    print("📂 Reading pairs...")
    pairs = read_pairs(LFW_ROOT, MATCH_FILE, MISMATCH_FILE)
    print(f"✅ Found {len(pairs)} pairs")

    # Load Model (ใช้ model เดียวกันทั้งคู่)
    model = load_model(CHECKPOINT)
    print("✅ Using GPU:", torch.cuda.get_device_name(0))
    print("✅ Model Loaded!")

    # ==================================
    # [1/2] Baseline (FP32, Single)
    # ==================================
    print("\n" + "="*55)
    print("  [1/2] Baseline (FP32, Single Image)")
    print("="*55)
    results_baseline = evaluate_baseline(model, pairs)
    metrics_baseline = compute_biometric_metrics(
        results_baseline['similarities'],
        results_baseline['labels']
    )
    print_results("Baseline (FP32 | Single)",
                  results_baseline, metrics_baseline)

    # ==================================
    # [2/2] Advanced (FP32 + Batch)
    # ==================================
    print("\n" + "="*55)
    print(f"  [2/2] Advanced (FP32 + Batch={BATCH_SIZE})")
    print("="*55)
    results_advanced = evaluate_batch(model, pairs, batch_size=BATCH_SIZE)
    metrics_advanced = compute_biometric_metrics(
        results_advanced['similarities'],
        results_advanced['labels']
    )
    print_results(f"Advanced (FP32 | Batch={BATCH_SIZE})",
                  results_advanced, metrics_advanced)

    # ==================================
    # Comparison Summary
    # ==================================
    t_base  = results_baseline['timing']
    t_adv   = results_advanced['timing']
    speedup  = t_adv['throughput'] / t_base['throughput']
    acc_diff = (results_advanced['accuracy'] -
                results_baseline['accuracy']) * 100
    eer_diff = (metrics_advanced['EER'] -
                metrics_baseline['EER']) * 100

    print(f"\n{'='*55}")
    print("  📊 Comparison Summary")
    print(f"{'='*55}")
    print(f"  {'Metric':<25} {'Baseline':>10} {'Advanced':>10} {'Diff':>8}")
    print(f"{'-'*55}")
    print(f"  {'Accuracy (%)':<25} "
          f"{results_baseline['accuracy']*100:>10.2f} "
          f"{results_advanced['accuracy']*100:>10.2f} "
          f"{acc_diff:>+8.2f}")
    print(f"  {'EER (%)':<25} "
          f"{metrics_baseline['EER']*100:>10.2f} "
          f"{metrics_advanced['EER']*100:>10.2f} "
          f"{eer_diff:>+8.2f}")
    print(f"  {'AUC':<25} "
          f"{metrics_baseline['AUC']:>10.4f} "
          f"{metrics_advanced['AUC']:>10.4f} "
          f"{metrics_advanced['AUC']-metrics_baseline['AUC']:>+8.4f}")
    print(f"  {'Throughput (pairs/s)':<25} "
          f"{t_base['throughput']:>10.2f} "
          f"{t_adv['throughput']:>10.2f} "
          f"{speedup:>+8.2f}x")
    print(f"  {'Avg Inference (ms)':<25} "
          f"{t_base['avg_inference_ms']:>10.2f} "
          f"{t_adv['avg_inference_ms']:>10.2f} "
          f"{t_adv['avg_inference_ms']-t_base['avg_inference_ms']:>+8.2f}")
    print(f"{'='*55}")
    print(f"\n  🚀 Speedup    : {speedup:.2f}x faster")
    print(f"  📉 Acc change : {acc_diff:+.2f}%")
    print(f"  📉 EER change : {eer_diff:+.2f}%")
    print(f"{'='*55}")