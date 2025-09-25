# AI Skill Planner

A comprehensive platform for analyzing team skill gaps, capacity planning, and strategic workforce development in AI/ML organizations.

## 🎯 Project Objective

The AI Skill Planner addresses the critical challenge of skill gap identification and capacity planning in rapidly evolving AI/ML organizations. As technology advances and project requirements become increasingly complex, organizations need sophisticated tools to:

- **Identify skill gaps** before they become project bottlenecks
- **Optimize resource allocation** across multiple AI/ML projects
- **Plan strategic workforce development** initiatives
- **Minimize project delays** caused by skill shortages
- **Make data-driven hiring and training decisions**

## 🚀 Key Capabilities

### 📊 Executive Dashboard
- **Role-based access control** with tailored views for Admin, Executive, Manager, and Analyst roles
- **Real-time skill gap visualization** with interactive heat maps
- **Financial impact analysis** including NPV calculations and cost-of-delay modeling
- **Risk assessment** with Monte Carlo simulations for project outcomes
- **Automated recommendations** for hiring, training, and resource allocation

### 🔍 Advanced Analytics
- **Gap Analysis Engine**: Identifies skill shortages across projects and phases
- **Proficiency Calculator**: Tracks skill decay and updates effective skill levels
- **Capacity Model**: Analyzes team utilization and availability
- **Financial Model**: Calculates business impact of skill gaps and interventions
- **Validation Framework**: Ensures model accuracy with ground truth tracking

### 🏢 Production-Ready Features
- **Excel File Upload**: Bulk data import with comprehensive validation
- **Project Onboarding**: 4-step wizard for new project creation
- **User Management**: Role-based project assignments and access control
- **Automatic Recalculation**: Background processing for metrics updates
- **API Security**: JWT authentication with permission-based endpoints

### 📈 Business Intelligence
- **Heat Map Visualization**: Interactive skill gap matrix across projects
- **Top Gaps Analysis**: Prioritized list of critical skill shortages
- **Organization Overview**: High-level metrics and KPIs
- **Evidence Tracking**: Skills validation and certification management
- **Assignment Optimization**: Smart resource allocation recommendations

## 🛠 Technology Stack

### Backend
- **FastAPI**: Modern, high-performance web framework
- **SQLite**: Embedded database with full SQL support
- **Pydantic**: Data validation and settings management
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and Monte Carlo simulations

### Frontend
- **Vanilla JavaScript**: No framework dependencies for maximum performance
- **Chart.js**: Interactive charts and visualizations
- **D3.js**: Advanced data visualization for heat maps
- **CSS Grid/Flexbox**: Responsive, modern UI design

### Infrastructure
- **Docker**: Containerized deployment
- **Uvicorn**: ASGI server for production
- **Conda**: Environment management
- **Make**: Build automation and task management

## 📁 Project Structure

```
ai_skill_planner/
├── api/                          # FastAPI application
│   ├── main.py                   # Main application entry point
│   ├── admin_endpoints.py        # Admin-only endpoints (data upload, user management)
│   ├── demo_auth.py             # Simplified authentication for demo
│   ├── executive_endpoints.py   # Executive dashboard endpoints
│   ├── gap_endpoints.py         # Skill gap analysis endpoints
│   ├── metrics_recalculation.py # Automatic metrics updating
│   ├── user_project_endpoints.py # User-project assignment management
│   └── validation_endpoints.py  # Model validation endpoints
├── database/                     # Database management
│   ├── init_db.py               # Database initialization
│   ├── schema.sql               # Database schema definition
│   └── ai_skill_planner.db      # SQLite database file
├── engines/                      # Core analytics engines
│   ├── gap_analysis.py          # Skill gap detection and analysis
│   ├── proficiency.py           # Skill level calculation with decay
│   ├── capacity.py              # Team capacity and utilization analysis
│   ├── financial.py             # NPV and financial impact modeling
│   ├── risk_assessment.py       # Monte Carlo risk analysis
│   ├── recommendations.py       # AI-powered recommendations
│   └── validation.py            # Model accuracy validation
├── data/                         # Sample data generation
│   ├── skills_taxonomy.py       # AI/ML skills taxonomy
│   ├── generate_people.py       # Sample team data
│   ├── generate_projects.py     # Sample project data
│   └── generate_assignments.py  # Sample assignments
├── static/                       # Frontend assets
│   ├── executive_dashboard.html  # Main dashboard interface
│   └── heatmap.html             # Skill gap heat map
├── security/                     # Security and authentication
│   ├── auth.py                  # JWT authentication system
│   └── encryption.py           # Data encryption utilities
├── scripts/                      # Deployment and setup
│   └── setup_environment.sh    # Environment setup script
├── docker-compose.yml           # Docker orchestration
├── Dockerfile                   # Container configuration
├── Makefile                     # Build and deployment tasks
├── requirements.txt             # Python dependencies
└── environment.yml              # Conda environment
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker (optional)
- Make (optional)

### Option 1: Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai_skill_planner
   ```

2. **Set up environment**
   ```bash
   # Using conda (recommended)
   conda env create -f environment.yml
   conda activate ai_skill_planner

   # Or using pip
   pip install -r requirements.txt
   ```

3. **Initialize database**
   ```bash
   python database/init_db.py
   ```

4. **Generate sample data**
   ```bash
   python data/skills_taxonomy.py
   python data/generate_people.py
   python data/generate_projects.py
   python data/generate_assignments.py
   python data/generate_evidence.py
   ```

5. **Start the application**
   ```bash
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Option 2: Docker Deployment

1. **Using Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Using Make (if available)**
   ```bash
   make run
   ```

### Access the Application

Open your browser and navigate to: [http://localhost:8000](http://localhost:8000)

## 🔐 Demo Authentication

The system includes demo authentication with the following users:

| Role | Username | Password | Access Level |
|------|----------|----------|--------------|
| System Administrator | `admin` | `admin123` | Full system access, user management, data upload |
| Chief Technology Officer | `executive` | `exec123` | All projects, executive dashboards |
| Engineering Manager | `manager` | `manager123` | Assigned projects only |
| Data Analyst | `analyst` | `analyst123` | Assigned projects only |

## 📊 Key Features Walkthrough

### 1. Role-Based Dashboard
- Different navigation and functionality based on user role
- Executives see organization-wide metrics
- Managers see only their assigned projects
- Analysts have read-only access to relevant data

### 2. Skill Gap Heat Map
- Interactive matrix showing skill gaps across projects
- Color-coded severity levels (Critical, High, Medium, Low)
- Click-through for detailed gap analysis
- Export capabilities for further analysis

### 3. Admin Panel
- **Data Upload**: Excel file import with validation
- **Project Onboarding**: Guided wizard for new projects
- **User Management**: Assign users to projects
- **System Monitoring**: View system metrics and activity

### 4. Financial Impact Analysis
- **NPV Calculations**: Net present value of training vs. hiring
- **Cost of Delay**: Impact of skill gaps on project timelines
- **ROI Analysis**: Return on investment for skill development
- **Monte Carlo Simulations**: Risk assessment for different scenarios

### 5. Automated Recommendations
- **Hiring Recommendations**: When to hire vs. train
- **Training Priorities**: Which skills to develop first
- **Resource Allocation**: Optimal team assignments
- **Risk Mitigation**: Strategies to reduce project risk

## 🔧 API Documentation

### Authentication Endpoints
- `POST /auth/login` - User authentication
- `POST /auth/logout` - User logout
- `GET /auth/me` - Current user information

### Core Data Endpoints
- `GET /skills` - List all skills with optional filtering
- `GET /people` - List team members with skill profiles
- `GET /projects` - List projects with role-based filtering
- `GET /assignments` - Project assignments and allocations

### Analytics Endpoints
- `GET /gap-analysis/organization/overview` - Organization-wide gap analysis
- `GET /gap-analysis/project/{id}/gaps` - Project-specific gaps
- `GET /gap-analysis/heatmap` - Heat map data
- `GET /gap-analysis/top-gaps` - Prioritized gap list

### Executive Endpoints
- `GET /executive/dashboard` - Executive summary metrics
- `GET /executive/financial/overview` - Financial impact analysis
- `GET /executive/recommendations/organization` - Strategic recommendations
- `GET /executive/risk/organization` - Risk assessment summary

### Admin Endpoints
- `POST /admin/upload/project` - Excel project data upload
- `POST /admin/upload/team` - Excel team data upload
- `POST /admin/onboard/project` - New project onboarding
- `GET /admin/system/metrics` - System performance metrics

### User-Project Management
- `GET /user-projects/user/{user_id}/projects` - Projects assigned to user
- `POST /user-projects/assign` - Assign user to project
- `GET /user-projects/assignments/summary` - Assignment overview

### Metrics Recalculation
- `GET /metrics/recalculation/status` - Current recalculation status
- `POST /metrics/recalculation/trigger` - Manually trigger recalculation
- `POST /metrics/demo/simulate-changes` - Simulate data changes

## 🧪 Data Model

### Core Entities
- **Skills**: AI/ML skills taxonomy with decay rates
- **People**: Team members with skill profiles and availability
- **Projects**: AI/ML projects with phases and requirements
- **Assignments**: Resource allocation across projects
- **Evidence**: Skills validation and certification tracking

### Key Relationships
- People have Skills with proficiency levels
- Projects require Skills at specific levels
- People are Assigned to Projects for specific phases
- Evidence validates Skills for People

## 📈 Analytics Models

### Gap Analysis
- **Coverage Ratio**: Available skills vs. required skills
- **Bus Factor**: Risk of key person dependencies
- **Severity Classification**: Critical, High, Medium, Low gaps
- **Cost Impact**: Weekly cost of delays due to gaps

### Proficiency Calculation
- **Base Level**: Initial skill assessment (0-5 scale)
- **Effective Level**: Adjusted for recency and evidence
- **Confidence Intervals**: Uncertainty bounds on assessments
- **Decay Modeling**: Time-based skill degradation

### Financial Modeling
- **NPV Analysis**: Training vs. hiring cost comparison
- **Break-even Analysis**: Time to positive ROI
- **Monte Carlo Risk**: Probabilistic outcome modeling
- **Sensitivity Analysis**: Impact of key assumptions

## 📊 Sample Data Included

The system comes pre-loaded with realistic test data:

- **42 AI/ML skills** across 8 categories (Machine Learning, MLOps, Data Engineering, etc.)
- **20 team members** with diverse experience levels and hourly rates ($65-180/hr)
- **4 AI projects** with realistic timelines and financial impact
- **95+ project assignments** with capacity-aware allocation
- **270+ evidence pieces** (certifications, deployments, incidents)

### Sample Projects
1. **Real-time Fraud Detection System V2** - High-value ML project
2. **E-commerce Personalization Engine** - Customer-facing AI system
3. **Manufacturing Quality Control Vision System** - Computer vision application
4. **AI-Powered Customer Support Platform** - NLP-based automation

## 🚀 Deployment

### Production Considerations
- **Database**: Replace SQLite with PostgreSQL for production
- **Authentication**: Integrate with corporate SSO (LDAP/SAML)
- **Monitoring**: Add logging, metrics, and alerting
- **Security**: Enable HTTPS and security headers
- **Scaling**: Use multiple workers and load balancing

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/dbname

# Authentication
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256

# API
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false
```

### Docker Production Build
```bash
# Build optimized production image
docker build -t ai-skill-planner:production .

# Run with environment variables
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e JWT_SECRET_KEY=... \
  ai-skill-planner:production
```

## 🧪 Testing

### API Health Check
```bash
# Check if API is running
curl http://localhost:8000/health

# Test authentication
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'
```

### Interactive API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔧 Development

### Adding New Features
1. **Backend**: Add endpoints in `api/` directory
2. **Analytics**: Implement engines in `engines/` directory
3. **Frontend**: Update `static/executive_dashboard.html`
4. **Database**: Modify schema in `database/schema.sql`

### Data Generation
```bash
# Regenerate sample data
python data/skills_taxonomy.py
python data/generate_people.py
python data/generate_projects.py
python data/generate_assignments.py
python data/generate_evidence.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check the API documentation at `/docs` when running the application
- Review the sample data generators for understanding the data model

## 🎉 Acknowledgments

- Built with modern web technologies for scalability and performance
- Designed for AI/ML organizations with complex skill requirements
- Validated with real-world use cases and industry feedback

---

**AI Skill Planner** - Empowering organizations to build the right teams for AI success.