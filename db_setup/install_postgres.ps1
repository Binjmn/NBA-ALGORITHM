<#
.SYNOPSIS
    PostgreSQL Installer and Configuration Script for NBA Prediction System
.DESCRIPTION
    This script installs and configures PostgreSQL for the NBA Prediction System.
    It provides a complete setup including:
    - PostgreSQL installation
    - Database creation
    - User setup with appropriate permissions
    - Verification of the installation
#>

# Script must be run as administrator
if (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "This script requires administrative privileges. Please run as Administrator."
    exit 1
}

$ProgressPreference = 'SilentlyContinue'  # Hide progress bars for faster execution

# Configuration
$PostgresVersion = "14.7"
$PostgresPath = "C:\Program Files\PostgreSQL\$PostgresVersion"
$PostgresDataPath = "$PostgresPath\data"
$PostgresUsername = "postgres"
$PostgresPassword = "ALGO123"
$DatabaseName = "nba_prediction"

# Start transcript for logging
Start-Transcript -Path "$PSScriptRoot\postgres_setup_log.txt" -Force

function Test-CommandExists {
    param ($command)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'stop'
    try { if (Get-Command $command) { $true } }
    catch { $false }
    finally { $ErrorActionPreference = $oldPreference }
}

function Install-PostgreSQL {
    Write-Host "\n============================================="
    Write-Host "INSTALLING POSTGRESQL $PostgresVersion" -ForegroundColor Cyan
    Write-Host "=============================================\n"

    # Check if PostgreSQL is already installed
    if (Test-Path $PostgresPath) {
        Write-Host "PostgreSQL appears to be already installed at $PostgresPath" -ForegroundColor Yellow
        $choice = Read-Host "Do you want to continue with the existing installation? (Y/N)"
        if ($choice -ne "Y" -and $choice -ne "y") {
            Write-Host "Installation aborted by user." -ForegroundColor Red
            exit 1
        }
        return
    }

    # Create temp directory
    $tempDir = "$env:TEMP\postgresql_setup"
    New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

    # Download the installer
    $installerUrl = "https://get.enterprisedb.com/postgresql/postgresql-$PostgresVersion-1-windows-x64.exe"
    $installerPath = "$tempDir\postgresql_installer.exe"
    
    Write-Host "Downloading PostgreSQL installer..." -ForegroundColor Green
    try {
        Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing
    }
    catch {
        Write-Host "Error downloading PostgreSQL installer: $_" -ForegroundColor Red
        Write-Host "Please download it manually from https://www.enterprisedb.com/downloads/postgres-postgresql-downloads" -ForegroundColor Yellow
        exit 1
    }

    # Install PostgreSQL silently
    Write-Host "Installing PostgreSQL..." -ForegroundColor Green
    $installArgs = "--mode unattended --superpassword $PostgresPassword --serverport 5432 --prefix \"$PostgresPath\" --datadir \"$PostgresDataPath\""
    Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait

    # Verify installation
    if (Test-Path $PostgresPath) {
        Write-Host "PostgreSQL installed successfully!" -ForegroundColor Green
    }
    else {
        Write-Host "PostgreSQL installation failed. Please check the logs." -ForegroundColor Red
        exit 1
    }

    # Add PostgreSQL bin to PATH
    $env:Path += ";$PostgresPath\bin"
    [Environment]::SetEnvironmentVariable("Path", $env:Path, [System.EnvironmentVariableTarget]::Machine)
}

function Configure-PostgreSQL {
    Write-Host "\n============================================="
    Write-Host "CONFIGURING POSTGRESQL" -ForegroundColor Cyan
    Write-Host "=============================================\n"

    # Wait for PostgreSQL to be fully operational
    Write-Host "Waiting for PostgreSQL service to start..." -ForegroundColor Green
    Start-Sleep -Seconds 5

    # Check if psql is available
    if (-not (Test-CommandExists "psql")) {
        $env:Path += ";$PostgresPath\bin"
    }

    # Create the database
    Write-Host "Creating database '$DatabaseName'..." -ForegroundColor Green
    $createDbCmd = "psql -U $PostgresUsername -c \"CREATE DATABASE $DatabaseName;\" -w"
    $Env:PGPASSWORD = $PostgresPassword
    
    try {
        Invoke-Expression $createDbCmd | Out-Null
        Write-Host "Database created successfully!" -ForegroundColor Green
    }
    catch {
        Write-Host "Error creating database: $_" -ForegroundColor Red
        # Check if database already exists
        $checkDbCmd = "psql -U $PostgresUsername -c \"SELECT 1 FROM pg_database WHERE datname = '$DatabaseName';\" -w"
        $result = Invoke-Expression $checkDbCmd
        if ($result -match "1 row") {
            Write-Host "Database already exists. Continuing..." -ForegroundColor Yellow
        }
        else {
            exit 1
        }
    }

    # Enable required extensions
    Write-Host "Enabling required PostgreSQL extensions..." -ForegroundColor Green
    $extensions = @("uuid-ossp", "pg_trgm", "btree_gin")
    
    foreach ($extension in $extensions) {
        $extensionCmd = "psql -U $PostgresUsername -d $DatabaseName -c \"CREATE EXTENSION IF NOT EXISTS $extension;\" -w"
        try {
            Invoke-Expression $extensionCmd | Out-Null
            Write-Host "Extension $extension enabled" -ForegroundColor Green
        }
        catch {
            Write-Host "Warning: Could not enable extension $extension: $_" -ForegroundColor Yellow
        }
    }
}

function Verify-Installation {
    Write-Host "\n============================================="
    Write-Host "VERIFYING POSTGRESQL INSTALLATION" -ForegroundColor Cyan
    Write-Host "=============================================\n"

    # Check PostgreSQL version
    try {
        $versionCmd = "psql -U $PostgresUsername -c \"SELECT version();\" -w"
        $Env:PGPASSWORD = $PostgresPassword
        $version = Invoke-Expression $versionCmd
        Write-Host "PostgreSQL version information:" -ForegroundColor Green
        Write-Host $version
    }
    catch {
        Write-Host "Error connecting to PostgreSQL: $_" -ForegroundColor Red
        exit 1
    }

    # Check if the database exists
    $checkDbCmd = "psql -U $PostgresUsername -c \"SELECT 1 FROM pg_database WHERE datname = '$DatabaseName';\" -w"
    $result = Invoke-Expression $checkDbCmd
    
    if ($result -match "1 row") {
        Write-Host "Database '$DatabaseName' exists!" -ForegroundColor Green
    }
    else {
        Write-Host "Database '$DatabaseName' does not exist!" -ForegroundColor Red
        exit 1
    }

    # Create a test table and insert data to verify
    $testTableCmd = "psql -U $PostgresUsername -d $DatabaseName -c \"CREATE TABLE IF NOT EXISTS test_table (id serial PRIMARY KEY, name VARCHAR(100));\" -w"
    $insertDataCmd = "psql -U $PostgresUsername -d $DatabaseName -c \"INSERT INTO test_table (name) VALUES ('test_data');\" -w"
    $verifyDataCmd = "psql -U $PostgresUsername -d $DatabaseName -c \"SELECT * FROM test_table;\" -w"
    $cleanupCmd = "psql -U $PostgresUsername -d $DatabaseName -c \"DROP TABLE test_table;\" -w"

    try {
        Invoke-Expression $testTableCmd | Out-Null
        Invoke-Expression $insertDataCmd | Out-Null
        $data = Invoke-Expression $verifyDataCmd
        Invoke-Expression $cleanupCmd | Out-Null

        if ($data -match "test_data") {
            Write-Host "Database read/write test passed!" -ForegroundColor Green
        }
        else {
            Write-Host "Database read/write test failed!" -ForegroundColor Red
            exit 1
        }
    }
    catch {
        Write-Host "Error during database test: $_" -ForegroundColor Red
        exit 1
    }
}

function Export-DatabaseConfig {
    Write-Host "\n============================================="
    Write-Host "EXPORTING DATABASE CONFIGURATION" -ForegroundColor Cyan
    Write-Host "=============================================\n"

    # Create .env file for the application
    $envFilePath = "$PSScriptRoot\..\database.env"
    $envContent = @"
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=$DatabaseName
POSTGRES_USER=$PostgresUsername
POSTGRES_PASSWORD=$PostgresPassword
"@

    Set-Content -Path $envFilePath -Value $envContent
    Write-Host "Database configuration exported to $envFilePath" -ForegroundColor Green

    # Set environment variables
    [Environment]::SetEnvironmentVariable("POSTGRES_HOST", "localhost", [System.EnvironmentVariableTarget]::User)
    [Environment]::SetEnvironmentVariable("POSTGRES_PORT", "5432", [System.EnvironmentVariableTarget]::User)
    [Environment]::SetEnvironmentVariable("POSTGRES_DB", $DatabaseName, [System.EnvironmentVariableTarget]::User)
    [Environment]::SetEnvironmentVariable("POSTGRES_USER", $PostgresUsername, [System.EnvironmentVariableTarget]::User)
    [Environment]::SetEnvironmentVariable("POSTGRES_PASSWORD", $PostgresPassword, [System.EnvironmentVariableTarget]::User)

    # Update current session
    $env:POSTGRES_HOST = "localhost"
    $env:POSTGRES_PORT = "5432"
    $env:POSTGRES_DB = $DatabaseName
    $env:POSTGRES_USER = $PostgresUsername
    $env:POSTGRES_PASSWORD = $PostgresPassword

    Write-Host "Environment variables set for the current session and user profile" -ForegroundColor Green
}

# Main execution flow
try {
    Write-Host "\n===============================================" -ForegroundColor Yellow
    Write-Host "NBA PREDICTION SYSTEM - POSTGRESQL SETUP" -ForegroundColor Yellow
    Write-Host "===============================================\n" -ForegroundColor Yellow

    # Create directory structure
    New-Item -ItemType Directory -Force -Path "$PSScriptRoot" | Out-Null

    Install-PostgreSQL
    Configure-PostgreSQL
    Verify-Installation
    Export-DatabaseConfig

    Write-Host "\n===============================================" -ForegroundColor Green
    Write-Host "POSTGRESQL SETUP COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "===============================================\n" -ForegroundColor Green
    Write-Host "Database Name: $DatabaseName"
    Write-Host "Username: $PostgresUsername"
    Write-Host "Password: $PostgresPassword"
    Write-Host "\nThe NBA Prediction System is now ready to use the database.\n"
}
catch {
    Write-Host "\n===============================================" -ForegroundColor Red
    Write-Host "POSTGRESQL SETUP FAILED!" -ForegroundColor Red
    Write-Host "===============================================\n" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "\nPlease check the log file for details.\n"
}
finally {
    Stop-Transcript
}
