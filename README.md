# Programming Collaboration Prediction and Development Partner Recommendation on GitHub
*(Dự đoán hợp tác lập trình và khuyến nghị đối tác phát triển dự án GitHub)*

![GitHub API](https://img.shields.io/badge/GitHub_API-Data_Source-181717?style=flat&logo=github)
![GraphQL](https://img.shields.io/badge/GraphQL-Querying-E10098?style=flat&logo=graphql)
![Python](https://img.shields.io/badge/Python-SNA_&_ML-3776AB?style=flat&logo=python)
![Graph Autoencoder](https://img.shields.io/badge/Graph_Autoencoder-Link_Prediction-9C27B0?style=flat)
![Link Prediction](https://img.shields.io/badge/Link_Prediction-Graph_Inference-BA68C8?style=flat)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Modeling-F9A825?style=flat&logo=tensorflow)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML_Framework-F7931E?style=flat&logo=scikit-learn)
![Neural Network](https://img.shields.io/badge/Neural_Network-(MLP)-FFB300?style=flat&logo=pytorch)

---

## Objective & Context *(Mục tiêu & Bối cảnh)*

* The growth of Open Source Software (OSS) has turned platforms like GitHub into complex social hubs where millions of developers connect and create. Collaboration is the backbone of OSS success.
    *(Sự phát triển của Phần mềm Nguồn mở (OSS) đã biến các nền tảng như GitHub thành những trung tâm xã hội phức tạp, nơi hàng triệu nhà phát triển kết nối và sáng tạo. Hợp tác là xương sống cho sự thành công trong môi trường OSS.)*
* However, GitHub's massive scale creates a paradox: while collaboration opportunities are nearly infinite, finding and connecting with the right partners (especially for newcomers) is increasingly difficult.
    *(Tuy nhiên, quy mô khổng lồ của GitHub tạo ra một nghịch lý: cơ hội hợp tác là gần như vô hạn, nhưng việc tìm kiếm và kết nối với những đối tác phù hợp (đặc biệt đối với người mới) lại ngày càng khó khăn.)*
* This research proposes an approach based on **Social Network Analysis (SNA)** and advanced machine learning, reframing this challenge as a **link prediction** problem to identify potential collaborations.
    *(Nghiên cứu này đề xuất một cách tiếp cận dựa trên **Phân tích Mạng Xã hội (SNA)** và kỹ thuật học máy tiên tiến, bằng cách định hình lại nhiệm vụ này thành bài toán **dự đoán liên kết** (link prediction) để xác định các liên kết hợp tác tiềm năng.)*

---

## Key Scientific Contributions *(Đóng góp Khoa học Chính)*

This research provides the following key scientific contributions:
*(Nghiên cứu đưa ra các đóng góp khoa học chính sau:)*

1.  **Collaboration Network Modeling:** Constructing a weighted graph from real-world GitHub data to capture the complex structure of collaborative relationships.
    *(**Mô hình hóa Mạng lưới Hợp tác:** Xây dựng một đồ thị có trọng số từ dữ liệu thực tế của GitHub để nắm bắt cấu trúc phức tạp của các mối quan Vhệ hợp tác.)*
2.  **Multi-dimensional Feature Engineering:** Designing a comprehensive feature set including local structural metrics, global centrality indices, and dyadic node-pair attributes.
    *(**Phát triển Hệ thống Tính năng Đa chiều:** Thiết kế và trích xuất một tập hợp tính năng toàn diện bao gồm các số liệu cấu trúc cục bộ, chỉ số độ trung tâm toàn cục và các thuộc tính dyadic.)*
3.  **Comparative Model Evaluation:** A rigorous comparison between traditional ML models (SVM, tree-based methods) and a modern Graph Autoencoder (GAE) architecture.
    *(**Đánh giá So sánh Hiệu suất Mô hình:** So sánh nghiêm ngặt giữa các mô hình học máy truyền thống (như SVM, các phương pháp dựa trên cây) và kiến trúc học sâu đồ thị hiện đại (Graph Autoencoder - GAE).)*
4.  **Methodological Framework Proposal:** Introducing a framework for building a practical Developer Partner Recommendation System based on the best-performing model.
    *(**Đề xuất Khung phương pháp:** Dựa trên mô hình hoạt động tốt nhất, giới thiệu khung phương pháp luận để xây dựng một Hệ thống Đề xuất Đối tác Phát triển.)*

---

## Methodology *(Phương pháp Luận)*

### 3.1. Data Collection & Graph Construction *(Thu thập Dữ liệu & Xây dựng Đồ thị)*

* **Seed Data:** 90 unique repositories selected from a combined list of the top 100 most-forked and 100 most-starred repos under the 'tensorflow' topic, collected via the GitHub GraphQL API.
    *(**Dữ liệu hạt giống:** 90 kho lưu trữ duy nhất được chọn từ danh sách kết hợp của 100 kho lưu trữ có nhiều fork nhất và 100 kho lưu trữ có nhiều sao nhất dưới chủ đề 'tensorflow', thu thập bằng API GraphQL.)*
* **Edge Definition:** A collaboration edge $(u, v)$ is created if both developers $u$ and $v$ contributed to at least one common repository, ensuring meaningful relationships.
    *(**Xây dựng Cạnh Hợp tác:** Một cạnh hợp tác (u, v) được tạo ra nếu cả hai nhà phát triển u và v đóng góp vào ít nhất một kho lưu trữ chung, đảm bảo các cạnh đại diện cho mối quan hệ có ý nghĩa.)*
* **Network:** The final output is an undirected, weighted graph $G = (V, E, W)$, where $V$ are developers, $E$ are collaborations, and $W$ is the collaboration strength.
    *(**Mạng lưới:** Dữ liệu đầu ra tạo thành một đồ thị vô hướng có trọng số G = (V, E, W), trong đó các nút V là nhà phát triển, E là tập hợp các cạnh hợp tác, và W là hàm trọng số.)*
* **Network Size:** The final collaboration network contains **1,048,503 edges** among **8,654 nodes** (users).
    *(**Kích thước Mạng lưới:** Mạng lưới hợp tác cuối cùng chứa **1.048.503 cạnh** giữa **8.654 nút** (người dùng).)*

### 3.2. Feature Engineering & Models *(Thiết kế Tính năng & Mô hình)*

Three primary groups of features were engineered for each node pair $(u, v)$:
*(Ba nhóm tính năng chính được sử dụng cho mỗi cặp nút (u, v):)*

1.  **Heuristic/Structural Features:** Common Neighbors, Jaccard Coefficient, Adamic-Adar Index, Resource Allocation.
    *(**Tính năng Dựa trên Cấu trúc/Ước lượng:** Ví dụ: Common Neighbors, Jaccard Coefficient, Adamic-Adar Index, và Resource Allocation.)*
2.  **Node & Centrality Features:** Degree, Clustering Coefficient, Closeness, Betweenness, Eigenvector Centrality, and PageRank.
    *(**Tính năng Dựa trên Nút & Độ Trung tâm:** Bao gồm Degree, Clustering Coefficient, Closeness, Betweenness, Eigenvector Centrality, và PageRank.)*
3.  **Dyadic Features:** Generated from node-level features (e.g., Sum, Product, or Absolute Difference of the PageRanks of $u$ and $v$).
    *(**Tính năng Dyadic:** Được tạo ra từ các tính năng cấp nút (ví dụ: Tổng, Tích, hoặc Hiệu tuyệt đối của PageRank của hai nút).)*

Models Compared: *(Các mô hình được So sánh:)*

* **Classical ML:** Logistic Regression (LR), Random Forest (RF), XGBoost, Support Vector Machine (SVM), MLP.
    *(**Học máy Cổ điển:** Logistic Regression (LR), Random Forest (RF), XGBoost, Support Vector Machine (SVM), và Multi-layer perceptron (MLP).)*
* **Graph Representation Learning:** **Graph Autoencoder (GAE)**. The GAE uses a two-layer Graph Convolutional Network (GCN) as an encoder to generate node embeddings and a simple inner-product decoder.
    *(**Học Biểu diễn Đồ thị:** **Graph Autoencoder (GAE)**. GAE sử dụng Mạng Tích chập Đồ thị (GCN) hai lớp làm bộ mã hóa để tạo ra các nhúng nút, và bộ giải mã tích trong đơn giản để tái tạo ma trận kề.)*

---

## Results & Performance Analysis *(Kết quả & Phân tích Hiệu suất)*

Evaluation was performed using **Area Under the ROC Curve (AUC)** and **Average Precision (AP)**.
*(Việc đánh giá được thực hiện chủ yếu bằng cách sử dụng **Area Under the ROC Curve (AUC)** và **Average Precision (AP)**.)*

### 4. Model Performance (Table IV) *(Hiệu suất Mô hình (Bảng IV))*

| Model | Validation AUC | Test AUC / AP |
| :--- | :--- | :--- |
| **Graph Autoencoder (GAE)** | **0.9990** | **0.9992 (AP)** |
| Logistic Regression | 0.9749 | 0.9748 |
| Neural Network (MLP) | 0.8693 | 0.8686 |
| Random Forest | 0.5943 | 0.5948 |
| SVM | 0.5018 | 0.5016 |
| XGBoost | 0.5000 | 0.5000 |
| Gradient Boosting | 0.4504 | 0.4498 |

**Model Conclusions:**
*(Kết luận về Mô hình:)*

* The **Graph Autoencoder (GAE)** achieved outstanding performance (AUC ≈ 0.9990; AP ≈ 0.9992), proving its ability to learn effective latent representations that capture high-order relational patterns.
    *(**Graph Autoencoder (GAE)** đạt hiệu suất vượt trội (AUC ≈ 0.9990; AP ≈ 0.9992), chứng minh khả năng học các biểu diễn tiềm ẩn hiệu quả, nắm bắt các mẫu quan hệ bậc cao.)*
* **Logistic Regression** was also strong (AUC ≈ 0.975), suggesting the core structural features contain powerful predictive signals.
    *(**Logistic Regression** cũng hoạt động mạnh (AUC ≈ 0.975), cho thấy các tính năng cấu trúc cốt lõi đã chứa các tín hiệu dự đoán mạnh mẽ.)*
* Tree-based models (RF, XGBoost) showed limited discriminatory power, with some performing no better than random guessing (AUC ≈ 0.5).
    *(Các mô hình dựa trên cây (Random Forest, Gradient Boosting, XGBoost) cho thấy khả năng phân biệt hạn chế, với một số mô hình hoạt động không tốt hơn dự đoán ngẫu nhiên (AUC gần 0.5).)*

---

##  Conclusion & Practical Implications
*(Kết luận & Ý nghĩa Thực tiễn)*

This research concludes that the combination of network analysis, feature engineering, and graph representation learning forms an effective model for predicting collaborative behavior in large-scale OSS environments.
*(Nghiên cứu kết luận rằng sự kết hợp giữa phân tích mạng lưới, kỹ thuật tính năng, và học biểu diễn đồ thị tạo thành một mô hình hiệu quả để hiểu và dự đoán hành vi hợp tác trong các môi trường mã nguồn mở quy mô lớn.)*

The results demonstrate the potential of GAE-based models to power partner recommendation systems on GitHub, fostering community growth and project innovation.
*(Kết quả chứng minh tiềm năng của các mô hình dựa trên đồ thị để tăng cường hệ thống đề xuất đối tác trên GitHub, hỗ trợ tăng trưởng cộng đồng và đổi mới dự án.)*

---

##  Authors *(Nhóm Thực hiện)*

**Students:** *(Sinh viên thực hiện)*  
- Hồ Gia Thành  
- Huỳnh Thái Linh  
- Trương Minh Khoa  

**Supervisor:** *(Giảng viên hướng dẫn)* *ThS. Lê Nhật Tùng*  
**University:** *(Trường)* Trường Đại học Công nghệ TP. Hồ Chí Minh — *Khoa học Dữ liệu*  
**Year:** *(Năm thực hiện)* 2025

---
