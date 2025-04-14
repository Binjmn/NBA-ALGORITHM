# NBA Betting Prediction System Rules

## Project-Specific Rules

1. **Real Data Integrity**: All data processing functions must validate the source is genuine and current, with no fallback to synthetic data under any circumstances.

2. **Time Zone Standardization**: All timestamps, logs, and scheduling must use Eastern Standard Time (EST) consistently throughout the codebase.

3. **Version Compatibility**: All dependencies must include specific version numbers in requirements.txt to ensure long-term compatibility.

4. **API Key Security**: API keys must never appear in code or logs; they should be stored in an encrypted configuration, loaded securely at runtime.

5. **Update Frequency**: Data collection and model updates must adhere strictly to the specified schedules (4-hour stats/odds, 10-minute live game data, etc.).

6. **Documentation Standards**: Each module must include standardized documentation with purpose, inputs, outputs, and example usage.

7. **Model Performance Thresholds**: Models must maintain minimum performance thresholds (e.g., 5% drift trigger) with automatic retraining when thresholds are crossed.

8. **JSON Output Format**: All prediction outputs must use a standard, consistent JSON format for compatibility with future UI/website integration.

9. **Pipeline Modularity**: Each component in the prediction pipeline must be independently testable and replaceable.

10. **Season-aware Design**: All code must be season-agnostic, automatically detecting and adapting to current and future NBA seasons.

## Global Rules

1. **Comprehensive Error Logging**: All errors must be logged with contextual information and appropriate severity levels.

2. **Scheduled Backup Protocol**: Database and critical model weights must be backed up daily with rolling retention policies.

3. **Resource Monitoring**: API usage and system resources must be continuously monitored to prevent exceeding rate limits or resource constraints.

4. **Code Review Standards**: All new code must include tests and meet minimum code coverage requirements before integration.

5. **Naming Convention**: Use clear, consistent naming patterns (snake_case for Python files/functions, PascalCase for classes) with descriptive names that indicate purpose.

## Code Quality Standards

1. **Code must be production-ready**: All code must include proper error handling, logging, documentation, and testing. No simplistic or prototype versions.

2. **Documentation must be modular**: README divided into distinct sections for clarity, and Architecture documentation separated into individual files for detailed structure.

3. **Version Control**: All edits should have their own descriptive commit to ensure we can revert to previous states.

4. **Best Practices**: All code should be written following best practices of the language being used.

5. **Latest Language Features**: When writing code, ensure you're using the latest version of the language.

## General Rules

1. **Language**: Speak to me in English.

2. **File Documentation**: When creating a new file, be sure its purpose and description is added to the README.md file.

3. **Function Documentation**: When creating a new function, be sure its purpose and description is added to the README.md file.

4. **Git Repository**: All projects should have their own git repository.

5. **Architecture Documentation**: All projects should have an architecture.md file that contains the file architecture of the project.

6. **Commit Practices**: All edits should have their own descriptive commit to ensure we can revert to the previous state at any point.

7. **Code Quality**: All code should be written in the best practice manner.

8. **Language Version**: When writing in a language, ensure you're using the latest version of the language.

9. **Changelog Consistency**: When adding or updating the changelog in the README.md file, make sure it's also updated in the main project file as well so the versions match.

10. **Code Authorization**: Never write code unless explicitly told to do so.

11. **Feature Development**: Never start fixing bugs or adding new features unless that's what is EXPLICITLY requested.

12. **Development Confirmation**: Always ask first before starting to code to double check that's what is wanted.

13. **Production Quality**: All code must be production quality with proper error handling, logging, documentation, and testing. Never implement simplistic or prototype versions that would require later rework for production use.

14. **Documentation Structure**: Documentation must always be modular and organized: README divided into distinct sections for clarity, and Architecture separated into individual files for detailed structure.

## File Organization

- `src/` - Source code
- `data/` - Data storage
- `logs/` - Application logs
- `docs/` - Documentation
- `config/` - Configuration files
- `tests/` - Unit and integration tests

## Changelog

### Version 1.0.0 (2025-04-14)
- Initial project rules established
