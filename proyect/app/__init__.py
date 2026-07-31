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

        # Precalentar el índice TF-IDF en un hilo daemon para que el servidor
        # quede listo sin esperar a la primera búsqueda del usuario.
        # Si existe caché en disco con el mismo doc_count lo carga en segundos;
        # si no, lo construye desde la DB y lo guarda para futuros reinicios.
        import threading as _threading

        def _warmup_tfidf():
            with app.app_context():
                try:
                    from app.utils import tfidf_service
                    tfidf_service.build_index()
                except Exception as exc:
                    print(f"[TF-IDF] Error en precalentamiento: {exc}", flush=True)

        _threading.Thread(target=_warmup_tfidf, daemon=True, name="tfidf-warmup").start()

    return app