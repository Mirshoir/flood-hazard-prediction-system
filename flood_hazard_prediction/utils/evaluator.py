from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def display_metrics(y_true, y_pred):
    """
    Display classification metrics and confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    st.write(f"**Accuracy:** {acc:.3f}")
    st.write(f"**Precision:** {prec:.3f}")
    st.write(f"**Recall:** {rec:.3f}")
    st.write(f"**F1 Score:** {f1:.3f}")

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(set(y_true))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
