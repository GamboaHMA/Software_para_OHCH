from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "dev"

    from app.documents.routes import bp as documents_bp
    app.register_blueprint(documents_bp)

    return app