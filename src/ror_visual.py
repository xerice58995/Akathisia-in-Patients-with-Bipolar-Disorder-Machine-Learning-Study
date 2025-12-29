import matplotlib.pyplot as plt

# --- 數據準備 ---
# 這些數據來自您兩次跑出來的結果
data = {
    "Time Period": ["All Time (2004-2024)", "Recent (2019-2024)"],
    "ROR": [0.6029, 0.6892],
    "CI_Lower": [0.5247, 0.5829],
    "CI_Upper": [0.6928, 0.8147],
    "P_Value": ["< 0.001", "< 0.001"],
}

y_pos = [0, 1]  # Y軸位置

plt.figure(figsize=(10, 5))

# 1. 繪製誤差線 (Error Bars)
# xerr 需要分別給出 [左邊長度, 右邊長度]
xerr_lower = [data["ROR"][i] - data["CI_Lower"][i] for i in range(2)]
xerr_upper = [data["CI_Upper"][i] - data["ROR"][i] for i in range(2)]

plt.errorbar(
    x=data["ROR"],
    y=y_pos,
    xerr=[xerr_lower, xerr_upper],
    fmt="o",
    markersize=10,
    capsize=8,
    color="#2c3e50",
    label="ROR (95% CI)",
)

# 2. 繪製無效線 (ROR=1)
plt.axvline(
    x=1, color="red", linestyle="--", linewidth=1.5, label="No Difference (ROR=1)"
)

# 3. 版面美化
plt.yticks(y_pos, data["Time Period"], fontsize=12)
plt.xlabel("Reporting Odds Ratio (ROR) [Log Scale]", fontsize=12)
plt.title(
    "Comparison of Akathisia Risk: Aripiprazole vs. Brexpiprazole\n(Testing for Weber Effect)",
    fontsize=14,
    fontweight="bold",
)

# 設定 X 軸為對數座標 (流行病學標準畫法)
plt.xscale("log")
plt.xlim(0.2, 2.0)
plt.xticks([0.2, 0.5, 0.7, 1, 1.5, 2], ["0.2", "0.5", "0.7", "1", "1.5", "2"])

# 4. 加入數據標籤
for i in range(2):
    label = f"ROR: {data['ROR'][i]:.2f}\n(95% CI: {data['CI_Lower'][i]:.2f}-{data['CI_Upper'][i]:.2f})"
    plt.text(
        data["ROR"][i],
        y_pos[i] - 0.15,
        label,
        ha="center",
        va="top",
        fontsize=10,
        color="blue",
        fontweight="bold",
    )

# 5. 加入區域解釋
plt.text(
    0.3,
    1.5,
    "Favors Aripiprazole\n(Ari is Safer)",
    ha="center",
    color="green",
    fontsize=11,
    bbox=dict(facecolor="white", alpha=0.8),
)
plt.text(
    1.5,
    1.5,
    "Favors Brexpiprazole\n(Brex is Safer)",
    ha="center",
    color="red",
    fontsize=11,
    bbox=dict(facecolor="white", alpha=0.8),
)

plt.grid(axis="x", which="both", linestyle=":", alpha=0.5)
plt.legend(loc="lower right")
plt.tight_layout()

# 存檔與顯示
plt.savefig("Weber_Effect_Comparison.png", dpi=300)
plt.show()
