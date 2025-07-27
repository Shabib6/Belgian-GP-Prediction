# 🏁 Belgian GP 2025 — F1 Race Prediction Using Machine Learning

Welcome to my predictive analytics project for the **2025 Belgian Grand Prix at Circuit de Spa-Francorchamps**!  
This project combines real-world F1 data with machine learning to predict race outcomes **before lights out**.  

---

## 🧠 Overview

Formula 1 is more than speed — it's a dynamic sport driven by **data**, **strategy**, and **conditions**.  
In this project, I built a predictive model using:

- **Driver performances at Spa** (previous years)
- **Qualifying timings**
- **Sector performance**
- **Championship standings**
- **Weather conditions** (rain probability)

These inputs were used in a **Platt-scaled Gradient Boosting Classifier** to predict the most probable **podium finishers**.

---

## 📊 Technologies Used

- **Python** 🐍  
- **Pandas**, **NumPy** for data preprocessing  
- **Matplotlib** & **Seaborn** for visualizations  
- **Scikit-learn** for ML modeling  
- **Platt Scaling** for probabilistic calibration  
- **Jupyter Notebook** for experimentation

---

## 🏎️ Prediction Summary

- **🏆 1st Place: Max Verstappen**  
  Based on rain experience + Spa track dominance  
- **🥈 & 🥉: McLaren Drivers (Norris & Piastri)**  
  Sector consistency + qualifying pace

The model accounts for **rain variability** using a probabilistic adjustment to reduce confidence on uncertain driver outcomes.

---

## 🌧️ Weather Factor

Rain was expected during the race weekend, influencing:

- **Tire strategy**
- **Driver control**
- **Overtaking difficulty**

Drivers with limited rain racing data were penalized accordingly in prediction confidence.

---

## 📌 Why This Project?

- To merge my passion for **Formula 1** with **Machine Learning**  
- To showcase the real-world applications of **sports analytics**  
- To learn how uncertainty (like weather) can be modeled in predictive systems


## 📷 Sample Visuals

_Add screenshots or plots here_  
Example:
- Qualifying Time Comparison  
- Driver Rain-Adjusted Probability Graph  

---

## 🧠 Future Work

- Integrate **telemetry data** (braking, acceleration, corner speeds)  
- Model **safety car events** & **pit stop strategies**  
- Extend predictions to full race classification (P1–P20)  

---

## 🗣️ Let’s Connect!

If you're an F1 fan, ML nerd, or both – feel free to connect!  
📩 LinkedIn: [Syed Shabib Ahamed](https://www.linkedin.com/in/syed-shabib-ahamed-b673b0225/)

---

## ⭐ Give it a Star!

If you liked this project, consider giving it a ⭐ on GitHub. It motivates me to do more data+F1 projects!

---

#F1 #MachineLearning #SportsAnalytics #BelgianGP #Formula1 #DataScience #AI #PredictiveModeling
