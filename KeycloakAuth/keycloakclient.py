import requests
from os import environ, getenv
from sys import stdout
import logging
from flask import request, url_for, session
import jwt
from base64 import b64decode
from cryptography.hazmat.primitives import serialization

class KeycloakClient:
    def __init__(self):
        self.KEYCLOAK_BASE_URL = environ["KEYCLOAK_BASE_URL"]
        self.REALM_NAME = environ["realm"]
        
        # Auth Configuration
        self.OIDC_USERNAME_CLAIM = getenv("OIDC_USERNAME_CLAIM", "email")
        self.OIDC_ISSUER_URL = environ["OIDC_ISSUER_URL"]
        self.OIDC_CLIENT_ID = environ["OIDC_CLIENT_ID"]
        self.OIDC_CLIENT_SECRET = environ["OIDC_CLIENT_SECRET"]
        self.OIDC_AUTH_ENDPOINT_URL = f"{self.OIDC_ISSUER_URL}/protocol/openid-connect/auth"
        self.OIDC_TOKEN_ENDPOINT_URL = f"{self.OIDC_ISSUER_URL}/protocol/openid-connect/token"
        self.AUD = "account"
        self.BEARER_PREFIX = "Bearer "

        # Debug Configuration
        DEBUG = getenv("DEBUG", "false")
        self.logger = logging.getLogger(__name__)
        if DEBUG == "True" or DEBUG == "true":
            self.logger.addHandler(logging.StreamHandler(stdout))
            self.logger.setLevel(logging.DEBUG)
    
    def getToken(self, authorization_code, _redirect_uri):
        payload = {
            "code": authorization_code,
            "grant_type": "authorization_code",
            "redirect_uri": f"{_redirect_uri}",
            "client_id": f"{self.OIDC_CLIENT_ID}",
            "client_secret": f"{self.OIDC_CLIENT_SECRET}",
        }
        resp_token = requests.post(f"{self.OIDC_TOKEN_ENDPOINT_URL}", data=payload)
        # self.logger.debug(resp_token.json())
        session["token"] = resp_token.json()["access_token"]
        session["refresh_token"] = resp_token.json()["refresh_token"]

    def intropectToken(self, token) -> bool:
        OIDC_INTROSPECT_TOKEN_ENDPOINT_URL = f"{self.OIDC_TOKEN_ENDPOINT_URL}/introspect"
        payload = {
                "token": token,
                "client_id": f"{self.OIDC_CLIENT_ID}",
                "client_secret": f"{self.OIDC_CLIENT_SECRET}",
            }
        resp = requests.post(f"{OIDC_INTROSPECT_TOKEN_ENDPOINT_URL}", data=payload)
        if not resp.json()["active"]:
            return False
        return True

    def refreshExpiredToken(self):
        self.logger.debug("Refreshing the token -------- ")
        payload = {
                "refresh_token": session["refresh_token"],
                "grant_type": "refresh_token",
                "client_id": f"{self.OIDC_CLIENT_ID}",
                "client_secret": f"{self.OIDC_CLIENT_SECRET}",
            }
        resp_token = requests.post(f"{self.OIDC_TOKEN_ENDPOINT_URL}", data=payload)
        session["token"] = resp_token.json()["access_token"]
        session["refresh_token"] = resp_token.json()["refresh_token"]
    
    def getUserInfo(self, token):
        user_info = dict()
        user_info["username"] = token[self.OIDC_USERNAME_CLAIM]
        user_info["user_id"] = token["user_id"]
        # Handle no Permission here
        try:
            user_info["is_manager"] = token["resource_access"]["realm-management"]["roles"][0] == "manage-users"
        except KeyError:
            user_info["is_manager"] = False
        try:
            user_info["is_admin"] = token["jupyterhub"]["roles"][0] == "admin" 
        except KeyError:
            user_info["is_admin"] = False
        try:
            user_info["permissions"] = token["Permissions"] ## If the permission doesnt exist for user than use default permissions
        except:
            user_info["permissions"] ={"Experiments": {}, "RegisteredModels": {}}
        session["user_info"] = user_info

    def decodeToken(self, token):
        _issuer_req = requests.get(self.OIDC_ISSUER_URL)
        _public_key = serialization.load_der_public_key(b64decode(_issuer_req.json()["public_key"].encode()))   
        jwt_token = jwt.decode(token, _public_key, algorithms=['HS256', 'RS256'], audience=self.AUD, options={"verify_iat": False})
        return jwt_token
    
    def getUser(self, token):
        token = self.decodeToken(token)
        user_info = dict()
        user_info["username"] = token[self.OIDC_USERNAME_CLAIM]
        user_info["user_id"] = token["user_id"]
        # Handle no Permission here
        try:
            user_info["is_manager"] = token["resource_access"]["realm-management"]["roles"][0] == "manage-users"
        except KeyError:
            user_info["is_manager"] = False
        try:
            user_info["is_admin"] = token["jupyterhub"]["roles"][0] == "admin" 
        except KeyError:
            user_info["is_admin"] = False
        try:
            Permissions = token["Permissions"] ## If the permission doesnt exist for user than use default permissions
            experiment_permissions = [{"experiment_id": name, "permission": permission} for name, permission in Permissions["Experiments"].items()]
            registered_model_permissions = [{"name": name, "permission": permission} for name, permission in Permissions["RegisteredModels"].items()]
            user_info["experiment_permissions"] = experiment_permissions
            user_info["registered_model_permissions"] = registered_model_permissions
        except:
            user_info["experiment_permissions"] = []
            user_info["registered_model_permissions"] = []
        return user_info

    def createGroup(self, group_name:str, parentGroup:str, user_id:str):
        """ This function creates groups in Keycloak to handle user permissions for Experiment
            and registeredModels.
            :param :
                group_name: name of created Experiment/RegisteredModel
                access_token: token for Keycloak
                parentGroup: Experiments or RegisteredModels
                user_id: Current User_id
        """
        # Get Id of mlflow group
        url = f'{self.KEYCLOAK_BASE_URL}/admin/realms/{self.REALM_NAME}/groups?search={parentGroup}'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + session["token"]
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        parent_group_id = response.json()[0]['subGroups'][0]["id"]
        # Create SubGroup using the parent group id (here parent group is mlflow)
        current_group_id = self.creatSubGroup(group_name, parent_group_id)
        # Create READ, EDIT and MANAGE under the group 
        ## For READ
        self.creatSubGroup("READ", current_group_id)
        ## For EDIT
        self.creatSubGroup("EDIT", current_group_id)
        ## For MANAGE
        manage_group_id = self.creatSubGroup("MANAGE", current_group_id)
        # Add current user to the group
        self.addUserToTheGroup(user_id, manage_group_id)

        # Refresh Token and User info so that we can get new Permissions for user 
        self.refreshExpiredToken()
        self.getUserInfo(session["token"])
    

    def creatSubGroup(self, group_name, parent_group_id):
        url = f'{self.KEYCLOAK_BASE_URL}/admin/realms/{self.REALM_NAME}/groups/{parent_group_id}/children'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + session["token"]
        }
        data = {
            "name": group_name
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["id"]

    def addUserToTheGroup(self, user_id, group_id):
        url = f'{self.KEYCLOAK_BASE_URL}/admin/realms/{self.REALM_NAME}/users/{user_id}/groups/{group_id}'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + session["token"]
        }
        response = requests.put(url, headers=headers)
        response.raise_for_status()
