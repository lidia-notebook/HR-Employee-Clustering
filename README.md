# HR: Employee Retention

## Business Understanding
Employee retention refers to an organization’s ability to maintain a stable and engaged workforce by creating an environment where employees are motivated to stay and grow. Strong retention safeguards institutional knowledge, sustains productivity, and enhances the organization’s reputation as an employer of choice. It is driven by factors such as fair compensation, career development opportunities, supportive leadership, manageable workloads, and a sense of purpose.

<img width="684" height="379" alt="Screenshot 2025-09-21 at 16 30 40" src="https://github.com/user-attachments/assets/b663c42e-b91d-430c-a276-fd0c13462741" />

### Why Retention Matters More Than Hiring
Hiring new employees is significantly more costly than retaining current ones. Recruitment expenses, training, and the productivity loss during the onboarding phase often exceed 30–150% of the departing employee’s annual salary. Beyond these visible costs, organizations also lose valuable institutional knowledge, disrupt team dynamics, and risk lowered customer satisfaction—consequences that can take months or even years to fully recover from.

## Problem Statement
HR needs clarity on which employees are most valuable and most at risk of leaving. Without data-driven segmentation, HR risks spending inefficiently and potentially losing top performers.

Key questions:
1. Who are our top performers at high risk of resigning?
2. Who are our low performers at high risk of resigning (should we let them go or upgrade them)?

## Goals

### Business Goals
- Reduce unnecessary hiring by retaining the right people
- Improve performance continuity

### Analytics Goals
- Create meaningful employee clusters
- Build a resignation risk score to rank employees inside each cluster

### Expected Impacts
- Increased productivity and loyalty
- Stronger customer relationships
- Reduced stress and fewer sick days
- Higher employee retention
- Strong employer brand to attract top talent

Source: https://therecruitmentco.uk/5-benefits-kind-employees/

## Retention Strategies and Recommendations
To maintain strong retention, provide benefits and treatments tailored to each cluster.

Source: https://acumenconnections.com/10-best-employee-benefits-to-hire-and-retain-top-talent/

Examples:
- Retirement plans  
- Flexible schedules  
- Health, vision, dental  
- Health and wellness programs  
- Team bonding activities  
- Mental health support  
- More paid time off (PTO)  
- Food events  
- Continuing education and student loan assistance  
- Pet-friendly workplace  

## Analytical Approach
We use a two-layer framework:

### Layer A - Clustering (Unsupervised)
- Group employees with similar characteristics
- Example clusters: high performance & underpaid, stable & fairly paid, new hires
- Outcome: easy-to-understand segments with clear drivers

### Layer B - Risk Index
- Build a resignation risk score for each employee
- Combine features like satisfaction, compensation, overtime, and performance

## Metrics and Evaluation
- Cluster quality: silhouette score, elbow method
- Risk index quality: accuracy of ranking vs actual resignation

## Workflow
1. Data Preparation and EDA
2. Feature Engineering
3. Preprocessing Pipeline
4. Clustering
5. Risk Index Calculation
6. Performance x Risk Quadrant
7. Notebook Deliverable
8. Streamlit App

## Requirements and Deployment
**Tech stack**
- Python (pandas, scikit-learn, seaborn, matplotlib, plotly, joblib)
- Jupyter Notebook
- Streamlit

**Deliverables**
- Jupyter Notebook (exploration and modeling)
- [Tableau for Data Visualization  ](https://public.tableau.com/app/profile/lidia.priskila/viz/WorkforceInsight/Dashboard4)
- [Streamlit app for HR to know clustering and prediction of the employee ](https://hr-employee-clustering-prediction.streamlit.app/)
- Final report with recommendations

---
