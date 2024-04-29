from os import getenv
from sys import stdout
import logging
from typing import Union
import jwt
import uuid
from .keycloakclient import KeycloakClient

from flask import Response, make_response, session, request, url_for
from werkzeug.datastructures import Authorization

client = KeycloakClient()

_logger = logging.getLogger(__name__)
DEBUG = getenv("DEBUG", "false")
if DEBUG == "True" or DEBUG == "true":
    _logger.addHandler(logging.StreamHandler(stdout))
    _logger.setLevel(logging.DEBUG)

_redirect_uri = url_for('serve', _external=True)

def authenticate_request() -> Union[Authorization, Response]:
    """ 
        Checks if there is any user_info in session redirects to the Keycloak login page if session is empty.
        :returns: Union[Authorization, Response]
    """
    resp = make_response()
    resp.status_code = 401
    resp.set_data(
        "You are not authenticated. Please provide a valid JWT Bearer token with the request."
    )
    resp.headers["WWW-Authenticate"] = 'Bearer error="invalid_token"'

    if session.get("user_info", None) is not None:
        # Checks if user is logged in or not.
        _logger.debug("session.get(\"user_info\", None) is not None")
        if client.intropectToken(session["token"]):
            return Authorization(auth_type="jwt", data=session["user_info"], token=session["token"])
        try :
            client.refreshExpiredToken()
            jwt_token = client.decodeToken(session["token"])
            client.getUserInfo(jwt_token)
            return Authorization(auth_type="jwt", data=session["user_info"], token=session["token"])
        except KeyError:
            # Means Request came from mlflow client no refresh_token in header
            return resp
    
    if session.get("state", None) is None:
        session["state"] = str(uuid.uuid4())

    token = request.headers.get("Authorization", None)
    _logger.warning("token form here")
    _logger.warning(token)
    code = request.args.get('code', None)
    if token is None and code is None:
        #  Means user is not logged in redirect them to login page
        _logger.debug("token is None and code is None")
        session["user_info"] = None
        resp.status_code = 301
        resp.headers["Content-Type"] = "application/x-www-form-urlencoded"
        resp.location = (f"{client.OIDC_AUTH_ENDPOINT_URL}"
                    "?response_type=code"
                    "&scope=openid email profile"
                    f"&client_id={client.OIDC_CLIENT_ID}"
                    f"&redirect_uri={_redirect_uri}"
                    f"&state={session['state']}"
                    )
        return resp
    resp.status_code = 401
    resp.set_data(
        "You are not authenticated. Please provide a valid JWT Bearer token with the request."
    )
    resp.headers["WWW-Authenticate"] = 'Bearer error="invalid_token"'
    if token is not None:
        # Means Request came from mlflow client
        _logger.debug("token is not None and code is None")
        # _logger.debug(token)
        if token.startswith(client.BEARER_PREFIX) or token.startswith(client.BEARER_PREFIX.lower()):
            token = token[len(client.BEARER_PREFIX):]
        try:
            jwt_token = client.decodeToken(token)
            if not jwt_token:
                _logger.warning("No jwt_token returned")
                return resp
            client.getUserInfo(jwt_token)
            session["token"] = token
            return Authorization(auth_type="jwt", data=session["user_info"], token=session["token"])
        except jwt.exceptions.InvalidTokenError:
            return resp
        
    if code is not None and request.headers.get("Authorization", None) is None:
        # Follow Authorization code flow
        _logger.debug("code is not None and request.headers.get(\"Authorization\", None) is None")
        if session.get('state', None) != request.args.get('state', None):
            _logger.debug("State changed between request.")
            return resp

        client.getToken(code, _redirect_uri)
        jwt_token = client.decodeToken(session["token"])
        if not jwt_token:
            _logger.warning("No jwt_token returned")
            return resp
        
        client.getUserInfo(jwt_token)
        # _logger.debug(session["user_info"])
        return Authorization(auth_type="jwt", data=session["user_info"], token=session["token"])
    return resp