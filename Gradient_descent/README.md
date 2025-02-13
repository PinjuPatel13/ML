
# 🔥 **Mastering Gradient Descent in Python**  

🎯 A hands-on implementation of **Gradient Descent** for optimizing a **Linear Regression Model**!  

---

## 🚀 **Introduction**
Gradient Descent is the backbone of many machine learning algorithms. It helps optimize models by iteratively adjusting parameters (`m` and `b`) to minimize the **cost function**. This project provides a **simple yet effective** implementation to understand its working.  

---

## 🧠 **Concepts Covered**
✔️ Understanding **Gradient Descent Algorithm**  
✔️ Implementing **Linear Regression** from scratch  
✔️ Optimizing the cost function using **partial derivatives**  
✔️ Step-by-step updates of model parameters (`m` and `b`)  

---

## 📂 **Project Structure**
```
📦 Gradient_Descent_Project
 ┣ 📜 Gradient_descent.py  # Python script implementing gradient descent
 ┣ 📜 README.md            # Project documentation
 ┣ 📜 requirements.txt      # Required dependencies
 ┗ 📊 dataset.csv          # Sample dataset (if applicable)
```

---

## 🔢 **Understanding the Problem**
We aim to fit a straight line using **Gradient Descent** for a given dataset:  

| 🏠 Area (sq ft) | 🛏 Bedrooms | 🏗 Age (years) | 💰 Price ($) |
|------------|------------|------------|------------|
| 1000 | 2 | 10 | 150,000 |
| 1500 | 3 | 5  | 200,000 |
| 2000 | 3 | 15 | 250,000 |
| 2500 | 4 | 7  | 310,000 |

📌 Our goal is to **predict the price** of a new house based on its **features** using **Gradient Descent**.  

---

## 🛠 **Installation & Setup**
Make sure you have **Python 3.x** installed. Then, install the required libraries using:  

```bash
pip install numpy pandas matplotlib
```

---

## 🏗 **How It Works**
1️⃣ Initialize `m` and `b` to zero.  
2️⃣ Compute `y_predicted = mx + b`.  
3️⃣ Calculate the **cost function** using **Mean Squared Error (MSE)**:  
   \[
   J(m, b) = \frac{1}{n} \sum (y - \hat{y})^2
   \]  
4️⃣ Compute **gradients** (partial derivatives) for `m` and `b`.  
5️⃣ Update values iteratively using **learning rate (α)**:  
   \[
   m = m - \alpha \cdot \frac{\partial J}{\partial m}
   \]  
   \[
   b = b - \alpha \cdot \frac{\partial J}{\partial b}
   \]  
6️⃣ Repeat until **convergence** (cost function stabilizes).  

---

## 📜 **Python Code**
Here’s the core implementation of **Gradient Descent**:

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x, y, learning_rate=0.008, iterations=10000):
    m_curr = b_curr = 0  # Initialize parameters
    n = len(x)  # Number of data points
    cost_history = []

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum((y - y_predicted) ** 2)  # Mean Squared Error
        md = -(2/n) * sum(x * (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        m_curr -= learning_rate * md  # Update m
        b_curr -= learning_rate * bd  # Update b
        cost_history.append(cost)

        if i % 1000 == 0:
            print(f"Iteration {i}: m={m_curr:.4f}, b={b_curr:.4f}, Cost={cost:.4f}")

    return m_curr, b_curr, cost_history

# Sample Data
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

# Run Gradient Descent
m_final, b_final, cost_history = gradient_descent(x, y)

# Plot Cost Function
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")
plt.show()
```

---

## 📈 **Results & Observations**
✅ The model optimizes `m` and `b` values to minimize the **cost function**.  
✅ The cost function **decreases over time**, ensuring convergence.  
✅ **Plotting cost vs. iterations** shows a smooth decline (good learning rate).  

---

## 🔧 **Hyperparameters to Experiment With**
You can tweak the following parameters for **better model performance**:  

| Hyperparameter | Description | Suggested Values |
|---------------|-------------|------------------|
| Learning Rate (α) | Step size for parameter updates | `0.001 - 0.01` |
| Iterations | Number of times the algorithm updates `m` and `b` | `5000 - 50000` |
| Dataset Size | More data = better predictions | Varies |

---

## 🤝 **Contributing**
💡 Found a bug? Have an idea for improvement? Feel free to **fork and contribute**! 🚀  

---
