#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "F2app.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)

from django.db import migrations


class Migration(migrations.Migration):
    
    dependencies = [
        ('fakeapp', '0003_remove_intervieweedetails_average_score_and_more'),  # replace with your actual migration name
    ]
    
    operations = [
        # First, temporarily remove the constraints
        migrations.RunSQL(
            "PRAGMA foreign_keys = OFF;",
            "PRAGMA foreign_keys = ON;"
        ),
        
        # Drop the old table (backup the data if needed)
        migrations.RunSQL(
            # For SQLite, since it doesn't support column changes easily
            "DROP TABLE IF EXISTS fakeapp_intervieweeskill_backup; "
            "CREATE TABLE fakeapp_intervieweeskill_backup AS SELECT * FROM fakeapp_intervieweeskill;",
            # Reverse operation (no action needed)
            ""
        ),
        
        # Remove the old table
        migrations.RunSQL(
            "DROP TABLE fakeapp_intervieweeskill;",
            # Reverse operation
            "CREATE TABLE fakeapp_intervieweeskill AS SELECT * FROM fakeapp_intervieweeskill_backup;"
        ),
        
        # Turn foreign keys back on
        migrations.RunSQL(
            "PRAGMA foreign_keys = ON;",
            "PRAGMA foreign_keys = OFF;"
        ),
    ]

if __name__ == "__main__":
    main()

