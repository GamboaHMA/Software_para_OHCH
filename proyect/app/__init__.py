import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()


def create_app(config_object=None):
    app = Flask(__name__)

    from app.config import Config
    app.config.from_object(Config)

    if config_object:
        app.config.from_object(config_object)

    # Ensure data directories exist
    for folder in [
        app.config["PDF_DIR"],
        app.config["OCR_DIR"],
        app.config["METADATA_DIR"],
        app.config["DB_DIR"],
    ]:
        os.makedirs(folder, exist_ok=True)

    db.init_app(app)
    migrate.init_app(app, db)

    from app.documents.routes import bp as documents_bp
    app.register_blueprint(documents_bp)

    with app.app_context():
        from app import models  # noqa: F401
        db.create_all()

    return app