### MsgAI - Advanced SMS Classification and Insight Tool 🌐

![Build Status](https://img.shields.io/badge/Build-Stable-brightgreen)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Documentation](https://img.shields.io/badge/Docs-Available-blue)

![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformer-yellow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2-orange)

#### **Problem Statement**

Managing SMS messages effectively can be daunting due to their unstructured nature. Important information may be lost amidst spam, financial transactions can go untracked, and event organization becomes inefficient. This lack of structure creates significant challenges for users seeking to derive actionable insights.

---

#### **Solution**

💡 **MsgAI** revolutionizes SMS management with a powerful AI-driven platform that intelligently categorizes messages, extracts actionable data, and integrates financial analysis and scheduling tools. By leveraging cutting-edge machine learning and natural language processing, MsgAI ensures seamless organization and insight generation.

---

#### **Core Features**

| **Feature**                 | **Description**                                                                             |
| --------------------------- | ------------------------------------------------------------------------------------------- |
| 🔎 **SMS Classification**   | Categorizes messages into spam, events, financial transactions, and specialized categories. |
| 📊 **Financial Analysis**   | Tracks income and expenses with trends and personalized budgeting tips.                     |
| 🗓 **Calendar Integration** | Extracts and schedules event-related dates with reminders and navigation tools.             |
| 🔧 **Data Management**      | Cleans and pre-processes SMS data to deliver accurate, actionable insights.                 |

---

### **Key Features in Detail**

##### **1. Intelligent SMS Classification 🔍**

MsgAI categorizes incoming messages into the following groups:

- **Spam Detection & Important Messages:** Utilizes multi-output classification with **scikit-learn** to identify and filter spam messages.
- **Event Scheduling:** Extracts dates directly from message content to organize events seamlessly.
- **Financial Transactions:** Identifies and categorizes income and expenditure details for financial insights.
- **Specialized Categories** (Enhanced by Modern BERT for high accuracy):
  - 📚 Education
  - 💸 Financial
  - 🏛️ Government
  - ⚕️ Health
  - 📢 Promotions
  - 🛒 Services/Retail
  - 📱 Telecommunications
  - 🚁 Travel

##### **2. Financial Analysis 📊**

- Automatically distinguishes income and expenditure within SMS content.
- Provides actionable insights through visualized reports and trends.
- Delivers personalized budgeting tips to improve financial health.

##### **3. Calendar Integration 🗓**

- Extracts event-related dates from SMS messages.
- Generates and tracks event schedules with a user-friendly interface.
- Offers reminders and easy navigation for upcoming and past events.

##### **4. Insightful Data Management 🔧**

- Cleans and pre-processes SMS data for maximum accuracy.
- Delivers tailored insights and actionable recommendations.

---

#### **Technologies Used**

| Technology       | Purpose                                     |
| ---------------- | ------------------------------------------- |
| **Camle Tool**   | Efficient data pre-processing pipeline.     |
| **OpenAI**       | Contextual understanding for NLP.           |
| **Modern BERT**  | Cutting-edge classification accuracy.       |
| **Flask**        | Lightweight API and UI management.          |
| **scikit-learn** | Spam detection and priority classification. |
| **HuggingFace**  | Advanced transformer models for NLP tasks.  |

---

#### **Project Structure**

```
├── Images
│   ├── dates.png
│   ├── home.png
│   ├── msgai.png
│   └── transaction.png
├── NoteBook
│   ├── bert_v1.ipynb
│   ├── classify_v2.ipynb
│   ├── Data_Cleaning_CAMeL.ipynb
│   ├── financial_analysis.ipynb
│   ├── preprocessing.ipynb
│   └── spam_importance.ipynb
├── README.md
├── requirements.txt
├── Scripts
│   ├── app.py
│   ├── static
│   │   ├── script.js
│   │   └── styles.css
│   └── templates
│       └── index.html
└── UI
    ├── app.py
    ├── date_analysis.py
    ├── financial_analysis.py
    ├── main.py
    ├── predict_category.py
    ├── predict_spam_importance.py
    ├── run_date_analysis.py
    ├── run_financial_analysis.py
    ├── run_predictions.py
    ├── static
    │   ├── css
    │   │   └── styles.css
    │   ├── image
    │   │   ├── msgai.png
    │   └── js
    │       └── scripts.js
    └── templates
        ├── base.html
        ├── dates.html
        ├── messages.html
        ├── spam.html
        └── transactions.html
```

---

#### **Team**

| Member  |
| ------- |
| Najla   |
| Mana    |
| Ibrahem |
| Naife   |

---

#### **Pages Overview**

##### **1. Homepage 🏠**

Showcases MsgAI’s features through a modern, intuitive interface.

![Homepage](https://raw.githubusercontent.com/AI-bootcamp/capstone-project-team-2-1.git/main/Images/home.png)

- Highlights classification insights.
- Displays financial trends and analysis.
- Quick access to event schedules.

##### **2. Spam and Importance Management (messages.html) 📮**

Organizes messages into spam and important categories.

- Advanced filtering and search capabilities.
- Intuitive visualization of categorized messages.

##### **3. Financial Transactions (transactions.html) 💸**

Provides detailed financial insights without requiring ML models.

![Transaction Analysis](https://raw.githubusercontent.com/AI-bootcamp/capstone-project-team-2-1.git/main/Images/transaction.png)

- Automatically categorizes transactions into income and expenditure.
- Visualizes financial trends and patterns.

##### **4. Calendar & Event Management (dates.html) 🗓**

Simplifies event organization by extracting dates from messages.

![Date and Schedule Management](https://raw.githubusercontent.com/AI-bootcamp/capstone-project-team-2-1.git/main/Images/dates.png)

- Enables reminder creation for key events.
- Tracks historical and upcoming schedules effectively.

##### **5. Base Template (base.html) 🌐**

A reusable, responsive framework ensures:

- Seamless navigation across pages.
- A consistent design theme inspired by professional tools like Hugging Face’s interface.

---

#### **Installation**

1. Clone the repository:

   ```bash
   git clone https://github.com/AI-bootcamp/capstone-project-team-2-1.git
   cd capstone-project-team-2-1
   ```

2. Navigate to the project directory and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   cd UI
   python app.py
   ```

4. Access the tool at: `http://127.0.0.1:5000`

---

#### **Logo**

- The MsgAI logo symbolizes precision and simplicity.
- A professional, clean design reinforces the tool’s mission of streamlining SMS insights.

---

#### **Future Enhancements**

1. **🚀 Real-time SMS Processing:** Enable live processing through advanced APIs.
2. **📱 Mobile App:** Develop a mobile-first application for enhanced accessibility.

