"""模型评估指标：Macro F1、混淆矩阵、分类报告"""
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report


def compute_metrics(all_labels, all_preds, class_names):
    """
    计算多分类评估指标

    Args:
        all_labels: list[int] — 真实标签列表
        all_preds: list[int] — 预测标签列表
        class_names: list[str] — 类别名称列表

    Returns:
        dict: {f1, accuracy, confusion_matrix, report}
    """
    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)

    f1 = f1_score(labels_arr, preds_arr, average='macro')
    per_class_f1 = f1_score(labels_arr, preds_arr, average=None)
    accuracy = (preds_arr == labels_arr).mean() * 100
    cm = confusion_matrix(labels_arr, preds_arr)
    report = classification_report(
        labels_arr, preds_arr,
        target_names=class_names,
        digits=4,
    )

    # 构建 per-class F1 字典
    per_class_f1_dict = dict(zip(class_names, per_class_f1))

    return {
        "f1": f1,
        "per_class_f1": per_class_f1_dict,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "report": report,
    }


def print_metrics(metrics):
    """格式化打印评估指标"""
    print(f"\n{'='*50}")
    print(f"Macro F1 Score:  {metrics['f1']:.4f}")
    print(f"Accuracy:        {metrics['accuracy']:.2f}%")
    print(f"{'='*50}")
    print("Per-Class F1:")
    for cls_name, f1_val in metrics["per_class_f1"].items():
        print(f"  {cls_name:<15s}: {f1_val:.4f}")
    print(f"{'='*50}")
    print("Classification Report:")
    print(metrics["report"])
    print(f"{'='*50}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(f"{'='*50}\n")
