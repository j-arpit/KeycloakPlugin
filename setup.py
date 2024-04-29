from setuptools import setup, find_packages

setup(
    name='KeycloakAuth',
    version='1.0',
    author='Arpit Joshi',
    author_email='arpitjoshi333@gmail.com',
    packages=find_packages(),
    package_data={'': ['basic_auth.ini']},
    install_requires=[
        'Flask',
        'flask-oidc',
        'flask_cors',
        'mlflow',
        'pyjwt',
        'flask_session',
    ],
    entry_points={
        "mlflow.app":(
            "keycloak-auth=KeycloakAuth:create_app"
        ),
    },
)
