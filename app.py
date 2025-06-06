import tkinter
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import requests
import threading
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Body, status, Request, File, UploadFile, Form, BackgroundTasks, \
    Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr, Field
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
import sqlite3
import os
import webbrowser
import json
import time
import uuid
import shutil
from collections import defaultdict
import asyncio

# --- CONFIGURATION ---
DATABASE_URL = "ai_generator_v3.db"
SECRET_KEY = os.environ.get("SECRET_KEY", "YOUR_VERY_SECRET_KEY_FOR_JWT_V3_CHANGE_ME_NOW")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7

STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY", "YOUR_STABILITY_AI_API_KEY_HERE")
STABILITY_API_HOST = "https://api.stability.ai"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
OPENAI_API_HOST = "https://api.openai.com/v1"

STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "sk_test_YOUR_STRIPE_SECRET_KEY")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY", "pk_test_YOUR_STRIPE_PUBLISHABLE_KEY")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "whsec_YOUR_STRIPE_WEBHOOK_SECRET")
STRIPE_CUSTOMER_PORTAL_LINK = os.environ.get("STRIPE_CUSTOMER_PORTAL_LINK",
                                             "YOUR_STRIPE_CUSTOMER_PORTAL_CONFIG_LINK_HERE")

BASE_URL = "http://localhost:8000"

GENERATED_IMAGES_DIR = "generated_images_v3"
USER_UPLOADS_DIR = "user_uploads_v3"
AVATARS_DIR = os.path.join(USER_UPLOADS_DIR, "avatars")
IMG2IMG_BASE_DIR = os.path.join(USER_UPLOADS_DIR, "img2img_bases")
ICONS_DIR = "ui_icons"

for dir_path in [GENERATED_IMAGES_DIR, USER_UPLOADS_DIR, AVATARS_DIR, IMG2IMG_BASE_DIR, ICONS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# --- UI STYLE CONFIGURATION ---
STYLE_CONFIG = {
    "dark_mode": True,
    "theme_color": "blue",  # Options: "blue", "green", "dark-blue"

    # ======== DARK THEME ========
    "APP_BG_COLOR_DARK": "#1A1B2F",             # Dark Navy Blue
    "FRAME_BG_COLOR_DARK": "#23263A",           # Slightly lighter for frame contrast
    "INPUT_BG_COLOR_DARK": "#2E334E",           # Steady neutral for input boxes
    "BUTTON_COLOR_DARK": "#5865F2",             # Discord Blue
    "BUTTON_TEXT_COLOR_DARK": "#FFFFFF",        # White text
    "BUTTON_HOVER_COLOR_DARK": "#6B75F8",       # Lighter blue on hover
    "ACCENT_COLOR_DARK": "#00BFFF",             # Deep Sky Blue for highlights
    "TEXT_COLOR_DARK": "#FFFFFF",               # Main readable text
    "TEXT_DISABLED_COLOR_DARK": "#767991",      # Muted gray
    "SUCCESS_COLOR_DARK": "#4CAF50",            # Green for success messages
    "ERROR_COLOR_DARK": "#EF5350",              # Slightly softer red for alerts
    "BORDER_COLOR_DARK": "#3A3F5E",             # Subtle border

    # ======== LIGHT THEME ========
    "APP_BG_COLOR_LIGHT": "#F5F7FA",            # Light gray-blue background
    "FRAME_BG_COLOR_LIGHT": "#FFFFFF",          # Pure white frame
    "INPUT_BG_COLOR_LIGHT": "#E8EAED",          # Soft gray input fields
    "BUTTON_COLOR_LIGHT": "#1877F2",            # Facebook Blue
    "BUTTON_TEXT_COLOR_LIGHT": "#FFFFFF",       # White button text
    "BUTTON_HOVER_COLOR_LIGHT": "#166FE5",      # Slightly darker hover
    "ACCENT_COLOR_LIGHT": "#18A0FB",            # Accent blue
    "TEXT_COLOR_LIGHT": "#202124",              # Very dark gray for readability
    "TEXT_DISABLED_COLOR_LIGHT": "#A0A0A0",     # Muted gray for disabled
    "SUCCESS_COLOR_LIGHT": "#4CAF50",           # Standard green
    "ERROR_COLOR_LIGHT": "#D32F2F",             # Rich red
    "BORDER_COLOR_LIGHT": "#DADCE0",            # Neutral gray border

    # ======== Typography & Layout ========
    "FONT_FAMILY_MAIN": "Roboto",
    "FONT_FAMILY_HEADINGS": "Roboto Slab",
    "FONT_SIZE_SMALL": 11,
    "FONT_SIZE_NORMAL": 13,
    "FONT_SIZE_MEDIUM": 15,
    "FONT_SIZE_LARGE": 18,
    "FONT_SIZE_XLARGE": 24,
    "FONT_WEIGHT_NORMAL": "normal",
    "FONT_WEIGHT_BOLD": "bold",

    # ======== Spacing & Corners ========
    "CORNER_RADIUS": 8,
    "MAIN_PADX": 20,
    "MAIN_PADY": 20,
    "WIDGET_PADX": 10,
    "WIDGET_PADY": 8,
    "SECTION_PADY": 15,
}


def get_style(key, mode_override=None):
    mode = mode_override if mode_override else ctk.get_appearance_mode().lower()
    # FIX: Correctly handle "System" mode by defaulting to a predictable state (e.g., light)
    # instead of incorrectly forcing dark mode values.
    if mode == "system":
        mode = "light"

    styled_key_dark = f"{key}_DARK"
    styled_key_light = f"{key}_LIGHT"

    if mode == "dark" and styled_key_dark in STYLE_CONFIG:
        return STYLE_CONFIG[styled_key_dark]
    elif mode == "light" and styled_key_light in STYLE_CONFIG:
        return STYLE_CONFIG[styled_key_light]
    # Fallback to the base key if a mode-specific one isn't found
    return STYLE_CONFIG.get(key)


SUBSCRIPTION_PLANS = {
    "free": {"name": "Free", "type": "subscription", "price_monthly": 0, "image_credits_monthly": 10,
             "max_resolution": "512x512", "features": ["Basic generation (Stability V1.6)"], "stripe_price_id": None,
             "paypal_plan_id": None},
    "basic": {"name": "Basic", "type": "subscription", "price_monthly": 10,
              "stripe_price_id": "price_BASIC_MONTHLY_ID_EXAMPLE", "paypal_plan_id": "P-BASIC_PAYPAL_ID_EXAMPLE",
              "image_credits_monthly": 75, "max_resolution": "1024x1024",
              "features": ["Standard generation", "Access to more styles", "DALL-E 2 (Limited)"]},
    "premium": {"name": "Premium", "type": "subscription", "price_monthly": 25,
                "stripe_price_id": "price_PREMIUM_MONTHLY_ID_EXAMPLE", "paypal_plan_id": "P-PREMIUM_PAYPAL_ID_EXAMPLE",
                "image_credits_monthly": 250, "max_resolution": "2048x2048",
                "features": ["Advanced generation", "All styles", "SDXL Access", "DALL-E 3 Access", "Priority support",
                             "Basic Editing Tools"]},
    "pro": {"name": "Pro", "type": "subscription", "price_monthly": 50,
            "stripe_price_id": "price_PRO_MONTHLY_ID_EXAMPLE", "paypal_plan_id": "P-PRO_PAYPAL_ID_EXAMPLE",
            "image_credits_monthly": 600, "max_resolution": "4096x4096",
            "features": ["Ultimate generation", "All models", "Full Editing Suite", "API access (Soon)",
                         "Batch processing (Enhanced)"]},
    "credit_pack_50": {"name": "50 Credits Pack", "type": "one_time_purchase", "price": 5, "credits_awarded": 50,
                       "stripe_price_id": "price_CREDITPACK50_ID_EXAMPLE", "features": ["One-time credit top-up"],
                       "paypal_plan_id": None, "max_resolution": "N/A", "image_credits_monthly": 0},
    "credit_pack_150": {"name": "150 Credits Pack", "type": "one_time_purchase", "price": 12, "credits_awarded": 150,
                        "stripe_price_id": "price_CREDITPACK150_ID_EXAMPLE",
                        "features": ["One-time credit top-up, best value"], "paypal_plan_id": None,
                        "max_resolution": "N/A", "image_credits_monthly": 0},
}

RATE_LIMIT_MAX_REQUESTS = 200
RATE_LIMIT_WINDOW_SECONDS = 60
request_counts = defaultdict(lambda: {"count": 0, "timestamp": time.time()})


def get_db_connection(): conn = sqlite3.connect(DATABASE_URL); conn.row_factory = sqlite3.Row; return conn


def create_tables():
    conn = get_db_connection();
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, email TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL, email_verified BOOLEAN DEFAULT FALSE, avatar_filename TEXT,
        subscription_plan TEXT DEFAULT 'free', subscription_status TEXT DEFAULT 'inactive', 
        plan_expiry_date DATETIME, image_credits_remaining INTEGER DEFAULT 0, last_credit_reset_date DATETIME,
        stripe_customer_id TEXT, paypal_payer_id TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP )""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS image_generations (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, prompt TEXT NOT NULL, negative_prompt TEXT,
        engine_id TEXT, seed INTEGER, steps INTEGER, cfg_scale REAL, width INTEGER, height INTEGER,
        image_filename TEXT NOT NULL, base_image_filename TEXT, is_public BOOLEAN DEFAULT FALSE, 
        tags TEXT, rating INTEGER, generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) )""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS payments (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, payment_gateway TEXT NOT NULL,
        transaction_id TEXT UNIQUE NOT NULL, subscription_id_gateway TEXT, item_id TEXT NOT NULL, 
        amount REAL NOT NULL, currency TEXT NOT NULL, status TEXT NOT NULL, coupon_code TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) )""")
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS password_reset_tokens (id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL,token TEXT UNIQUE NOT NULL,expires_at DATETIME NOT NULL,created_at DATETIME DEFAULT CURRENT_TIMESTAMP,used BOOLEAN DEFAULT FALSE,FOREIGN KEY (user_id) REFERENCES users (id))""")
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS email_verification_tokens (id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL,token TEXT UNIQUE NOT NULL,expires_at DATETIME NOT NULL,created_at DATETIME DEFAULT CURRENT_TIMESTAMP,used BOOLEAN DEFAULT FALSE,FOREIGN KEY (user_id) REFERENCES users (id))""")
    conn.commit();
    conn.close()


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(p, h): return pwd_context.verify(p, h)


def get_password_hash(p): return pwd_context.hash(p)


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    exp = datetime.now(timezone.utc) + (
        expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": exp, "user_id": data.get("user_id")})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)):
    exc = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials",
                        headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise exc
    username: str = payload.get("sub")
    user_id_from_token: int = payload.get("user_id")
    if username is None or user_id_from_token is None: raise exc
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ? AND username = ?',
                        (user_id_from_token, username)).fetchone()
    conn.close()
    if user is None: raise exc
    return dict(user)


async def get_current_active_user(cu: dict = Depends(get_current_user)): return cu


def reset_user_credits_if_needed(user_id: int):
    conn = get_db_connection();
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    if not user: conn.close();return
    plan_name = user['subscription_plan'];
    plan_details = SUBSCRIPTION_PLANS.get(plan_name)
    if not plan_details or plan_details.get("type") != "subscription": conn.close();return
    now = datetime.now(timezone.utc);
    last_reset_str = user['last_credit_reset_date'];
    needs_reset = False
    if not last_reset_str:
        needs_reset = True
    else:
        last_reset = datetime.fromisoformat(last_reset_str);
        needs_reset = (now - last_reset).days >= 30
    if needs_reset and (user['subscription_status'] == 'active' or plan_name == 'free'):
        new_credits = plan_details['image_credits_monthly']
        conn.execute('UPDATE users SET image_credits_remaining = ?, last_credit_reset_date = ? WHERE id = ?',
                     (new_credits, now.isoformat(), user_id))
        conn.commit()
    conn.close()


async def send_email_stub(to: str, subj: str, body: str): print(
    f"--EMAIL STUB--\nTo: {to}\nSubj: {subj}\nBody:\n{body}\n--------------")


async def _check_ai_permissions_and_credits(user_id: int, required_feature: str | None = None):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    if not user: conn.close(); raise HTTPException(status_code=404, detail="User not found.")
    reset_user_credits_if_needed(user_id)
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()

    current_plan_details = SUBSCRIPTION_PLANS.get(user['subscription_plan'])
    if not current_plan_details: conn.close(); raise HTTPException(status_code=403, detail="Invalid user plan.")

    if required_feature and required_feature not in current_plan_details.get("features", []):
        conn.close();
        raise HTTPException(status_code=403, detail=f"Feature '{required_feature}' not available in your plan.")

    if (user['subscription_status'] != 'active' and user['subscription_plan'] != 'free') or user[
        'image_credits_remaining'] <= 0:
        conn.close();
        raise HTTPException(status_code=402, detail="Insufficient credits or inactive subscription.")

    plan_max_w, plan_max_h = map(int, current_plan_details['max_resolution'].split('x'))
    conn.close()
    return user, plan_max_w, plan_max_h


async def generate_image_dalle3(prompt: str, user_id: int, width: int = 1024, height: int = 1024,
                                quality: str = "standard", n: int = 1, style: str = "vivid"):
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE": raise HTTPException(status_code=503,
                                                                                               detail="DALL-E 3 service unavailable.")
    user_data, _, _ = await _check_ai_permissions_and_credits(user_id, "DALL-E 3 Access")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"}
    valid_sizes = ["1024x1024", "1792x1024", "1024x1792"];
    req_size = f"{width}x{height}"
    if req_size not in valid_sizes: width, height, req_size = 1024, 1024, "1024x1024"; print(
        f"DALL-E 3 size adjusted to {req_size}")
    payload = {"model": "dall-e-3", "prompt": prompt, "n": n, "size": req_size, "quality": quality, "style": style}

    print(f"MOCK DALL-E 3: User {user_id}, Prompt: {prompt[:30]}...")
    await asyncio.sleep(2)
    import base64;
    placeholder_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    img_content = base64.b64decode(placeholder_b64)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S%f");
    img_fname = f"user_{user_id}_dalle3_{ts}.png"
    img_path = os.path.join(GENERATED_IMAGES_DIR, img_fname)
    with open(img_path, "wb") as f:
        f.write(img_content)

    conn = get_db_connection()
    conn.execute('UPDATE users SET image_credits_remaining=image_credits_remaining-1 WHERE id=?', (user_id,))
    conn.execute(
        'INSERT INTO image_generations (user_id,prompt,engine_id,width,height,image_filename) VALUES (?,?,?,?,?,?)',
        (user_id, prompt, "dall-e-3", width, height, img_fname))
    conn.commit();
    conn.close()
    return {"filename": img_fname, "url": f"/generated_images/{img_fname}"}


async def generate_image_sdxl(prompt: str, user_id: int, neg_prompt: str = "", width: int = 1024, height: int = 1024,
                              steps: int = 25, cfg_scale: float = 7.0, seed: int = 0,
                              base_image_path: str | None = None, base_image_strength: float = 0.35):
    await _check_ai_permissions_and_credits(user_id, "SDXL Access")
    return await generate_image_stabilityai(prompt, user_id, neg_prompt, engine_id="stable-diffusion-xl-1024-v1-0",
                                            steps=steps, cfg_scale=cfg_scale, width=width, height=height, seed=seed,
                                            base_image_path=base_image_path, base_image_strength=base_image_strength)


async def generate_image_stabilityai(prompt: str, user_id: int, neg_prompt: str = "",
                                     engine_id: str = "stable-diffusion-v1-6", steps: int = 30, cfg_scale: float = 7.0,
                                     width: int = 512, height: int = 512, seed: int = 0,
                                     base_image_path: str | None = None, base_image_strength: float = 0.35):
    if not STABILITY_API_KEY or STABILITY_API_KEY == "YOUR_STABILITY_AI_API_KEY_HERE": raise HTTPException(
        status_code=503, detail="StabilityAI service unavailable.")
    user_data, plan_max_w, plan_max_h = await _check_ai_permissions_and_credits(user_id)
    if width > plan_max_w or height > plan_max_h: raise HTTPException(status_code=403,
                                                                      detail=f"Resolution {width}x{height} exceeds plan limit {plan_max_w}x{plan_max_h}.")

    api_base = f"{STABILITY_API_HOST}/v1/generation/{engine_id}";
    headers = {"Accept": "application/json", "Authorization": f"Bearer {STABILITY_API_KEY}"}
    form_data_dict = {"text_prompts[0][text]": prompt, "cfg_scale": cfg_scale, "samples": 1, "steps": steps,
                      "seed": seed if seed > 0 else 0}
    if neg_prompt: form_data_dict.update({"text_prompts[1][text]": neg_prompt, "text_prompts[1][weight]": -1.0})

    files = {};
    json_payload = None;
    form_data_multipart = {}
    if base_image_path and os.path.exists(base_image_path):
        api_endpoint = f"{api_base}/image-to-image"
        form_data_dict.update(
            {"init_image_mode": "IMAGE_STRENGTH", "image_strength": base_image_strength, "width": width,
             "height": height})
        files["init_image"] = (os.path.basename(base_image_path), open(base_image_path, 'rb'),
                               f'image/{os.path.splitext(base_image_path)[1].lstrip(".")}')
        form_data_multipart = {k: str(v) for k, v in form_data_dict.items()}
    else:
        api_endpoint = f"{api_base}/text-to-image";
        headers["Content-Type"] = "application/json"
        json_payload = {"text_prompts": [{"text": prompt}], "cfg_scale": cfg_scale, "height": height, "width": width,
                        "samples": 1, "steps": steps, "seed": seed if seed > 0 else 0}
        if neg_prompt: json_payload["text_prompts"].append({"text": neg_prompt, "weight": -1.0})

    try:
        resp = requests.post(api_endpoint, headers=headers, data=form_data_multipart if files else None,
                             json=json_payload if not files else None, files=files if files else None,
                             timeout=180 if files else 120)
        resp.raise_for_status();
        api_data = resp.json()
        img_b64 = api_data["artifacts"][0]["base64"]
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S%f");
        img_fname = f"user_{user_id}_{engine_id.replace('-', '_')}_{ts}.png"
        img_path = os.path.join(GENERATED_IMAGES_DIR, img_fname)
        import base64;
        open(img_path, "wb").write(base64.b64decode(img_b64))

        conn = get_db_connection()
        conn.execute('UPDATE users SET image_credits_remaining=image_credits_remaining-1 WHERE id=?', (user_id,))
        db_base_fname = os.path.basename(base_image_path) if base_image_path else None
        current_seed_val = form_data_dict.get("seed") if files else json_payload.get("seed")
        conn.execute(
            'INSERT INTO image_generations (user_id,prompt,negative_prompt,engine_id,seed,steps,cfg_scale,width,height,image_filename,base_image_filename) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
            (user_id, prompt, neg_prompt, engine_id, current_seed_val, steps, cfg_scale, width, height, img_fname,
             db_base_fname))
        conn.commit();
        conn.close()
        return {"filename": img_fname, "url": f"/generated_images/{img_fname}"}
    except requests.exceptions.RequestException as e:
        err_detail = f"Stability API Error: {e}";
        raise HTTPException(status_code=500, detail=err_detail)
    finally:
        if files.get("init_image"): files["init_image"][1].close()


async def create_stripe_checkout_session(user_id: int, item_id: str,
                                         coupon_code: str | None = None):
    if not STRIPE_SECRET_KEY or STRIPE_SECRET_KEY.startswith("sk_test_YOUR"): raise HTTPException(status_code=503,
                                                                                                  detail="Stripe not configured.")
    import stripe;
    stripe.api_key = STRIPE_SECRET_KEY
    conn = get_db_connection();
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone();
    conn.close()
    if not user: raise HTTPException(status_code=404, detail="User not found.")
    item = SUBSCRIPTION_PLANS.get(item_id)
    if not item or not item.get("stripe_price_id"): raise HTTPException(status_code=400,
                                                                        detail="Invalid item or Stripe Price ID missing.")

    params = {'payment_method_types': ['card'], 'line_items': [{'price': item['stripe_price_id'], 'quantity': 1}],
              'mode': 'subscription' if item['type'] == 'subscription' else 'payment',
              'success_url': f"{BASE_URL}/payment-success?session_id={{CHECKOUT_SESSION_ID}}",
              'cancel_url': f"{BASE_URL}/payment-cancelled", 'metadata': {'user_id': str(user_id), 'item_id': item_id}}
    if user['stripe_customer_id']:
        params['customer'] = user['stripe_customer_id']
    else:
        params['customer_email'] = user['email'];
        params['customer_creation'] = 'always'
    if item['type'] == 'subscription': params['subscription_data'] = {
        'metadata': {'user_id': str(user_id), 'item_id': item_id}}
    try:
        session = stripe.checkout.Session.create(**params);
        return {"sessionId": session.id, "url": session.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stripe Error: {e}")


async def handle_stripe_webhook(request: Request):
    print("Stripe webhook received (mock processing)")
    return {"status": "success"}


async def update_user_subscription(user_id: int, plan_id: str, status: str, gateway: str, transaction_id: str,
                                   customer_id_gateway: str | None = None, subscription_id_gateway: str | None = None,
                                   conn_override=None):
    use_conn = conn_override if conn_override else get_db_connection()
    plan_details = SUBSCRIPTION_PLANS.get(plan_id)
    if not plan_details or plan_details.get("type") != "subscription": print(
        f"Error: Invalid plan_id {plan_id} for subscription.");return

    new_expiry_iso = (datetime.now(timezone.utc) + timedelta(days=31)).isoformat() if status == 'active' else None
    new_credits = plan_details['image_credits_monthly'] if status == 'active' else 0
    new_reset_date_iso = datetime.now(timezone.utc).isoformat()

    try:
        cursor = use_conn.cursor()
        cursor.execute(
            'UPDATE users SET subscription_plan=?,subscription_status=?,plan_expiry_date=?,image_credits_remaining=?,last_credit_reset_date=? WHERE id=?',
            (plan_id, status, new_expiry_iso, new_credits, new_reset_date_iso, user_id))
        if gateway == 'stripe' and customer_id_gateway: cursor.execute(
            'UPDATE users SET stripe_customer_id=? WHERE id=? AND stripe_customer_id IS NULL',
            (customer_id_gateway, user_id))
        cursor.execute(
            'INSERT INTO payments (user_id,payment_gateway,transaction_id,subscription_id_gateway,item_id,amount,currency,status,updated_at) VALUES (?,?,?,?,?,?,?,?,?)',
            (user_id, gateway, transaction_id, subscription_id_gateway, plan_id, plan_details['price_monthly'], "USD",
             status, datetime.now(timezone.utc).isoformat()))
        if not conn_override: use_conn.commit()
    except Exception as e:
        print(f"DB Error updating subscription for user {user_id}: {e}")
        if not conn_override:
            use_conn.rollback()
    finally:
        if not conn_override:
            use_conn.close()


fastapi_app = FastAPI(title="AI Generator API V3", version="3.0.0")
fastapi_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                           allow_headers=["*"])
fastapi_app.mount("/generated_images", StaticFiles(directory=GENERATED_IMAGES_DIR), name="generated_images")
fastapi_app.mount("/user_uploads", StaticFiles(directory=USER_UPLOADS_DIR), name="user_uploads")
fastapi_app.mount(f"/{ICONS_DIR}", StaticFiles(directory=ICONS_DIR), name="ui_icons")


async def rate_limiter_dependency(req: Request):
    return True


class UserCreate(BaseModel): username: str; email: EmailStr; password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    email_verified: bool
    avatar_url: str | None
    subscription_plan: str
    subscription_status: str
    plan_expiry_date: datetime | None
    image_credits_remaining: int
    stripe_customer_id: str | None = None

    class Config:
        from_attributes = True


class Token(BaseModel): access_token: str;token_type: str


class PromptPayload(
    BaseModel): prompt: str;negative_prompt: str | None = None;engine_id: str = "stable-diffusion-v1-6";steps: int = Field(
    default=30, ge=10, le=150);cfg_scale: float = Field(default=7.0, ge=1.0, le=20.0);width: int = Field(default=512,
                                                                                                         ge=256,
                                                                                                         le=4096);height: int = Field(
    default=512, ge=256, le=4096);seed: int = Field(default=0, ge=0);num_images: int = Field(default=1, ge=1,
                                                                                             le=4);base_image_filename: str | None = None;image_strength: float | None = Field(
    default=0.35, ge=0.0, le=1.0)


class PaymentRequest(BaseModel): item_id: str;payment_provider: str;coupon_code: str | None = None


class PasswordResetRequest(BaseModel): email: EmailStr


class PasswordResetConfirm(BaseModel): token: str;new_password: str


class ImageUpdateRequest(BaseModel): is_public: bool | None = None;tags: str | None = None;rating: int | None = Field(
    default=None, ge=1, le=5)


@fastapi_app.post("/register", response_model=UserResponse)
async def register_user_endpoint(uc: UserCreate, bg: BackgroundTasks):
    conn = get_db_connection();
    if conn.execute('SELECT id FROM users WHERE username=? OR email=?',
                    (uc.username, uc.email)).fetchone(): conn.close();raise HTTPException(status_code=400,
                                                                                          detail="Username or email already registered")
    hp = get_password_hash(uc.password);
    fp = SUBSCRIPTION_PLANS['free'];
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO users (username,email,hashed_password,subscription_plan,subscription_status,image_credits_remaining,last_credit_reset_date,email_verified) VALUES (?,?,?,?,?,?,?,?)',
        (uc.username, uc.email, hp, 'free', 'active', fp['image_credits_monthly'],
         datetime.now(timezone.utc).isoformat(), False))
    uid = cur.lastrowid;
    conn.commit()
    vt = str(uuid.uuid4());
    exp = datetime.now(timezone.utc) + timedelta(hours=24)
    cur.execute('INSERT INTO email_verification_tokens (user_id,token,expires_at) VALUES (?,?,?)',
                (uid, vt, exp.isoformat()));
    conn.commit()
    vl = f"{BASE_URL}/verify-email?token={vt}";
    bg.add_task(send_email_stub, uc.email, "Verify Email for AI Gen", f"Click to verify: {vl}")
    new_user_data = cur.execute('SELECT * FROM users WHERE id=?', (uid,)).fetchone();
    conn.close()

    user_data_dict = dict(new_user_data)
    if user_data_dict.get('avatar_filename'):
        user_data_dict['avatar_url'] = f"/user_uploads/avatars/{user_data_dict['avatar_filename']}"
    else:
        user_data_dict['avatar_url'] = None

    return UserResponse.model_validate(user_data_dict)


@fastapi_app.get("/verify-email")
async def verify_email_address_endpoint(token: str):
    conn = get_db_connection();
    td = conn.execute('SELECT * FROM email_verification_tokens WHERE token=? AND used=FALSE AND expires_at > ?',
                      (token, datetime.now(timezone.utc).isoformat())).fetchone()
    if not td: conn.close();raise HTTPException(status_code=400, detail="Invalid/expired token.")
    conn.execute('UPDATE users SET email_verified=TRUE WHERE id=?', (td['user_id'],));
    conn.execute('UPDATE email_verification_tokens SET used=TRUE WHERE id=?', (td['id'],));
    conn.commit();
    conn.close()
    return {"message": "Email verified. You can login."}


@fastapi_app.post("/token", response_model=Token)
async def login_for_access_token_endpoint(fd: OAuth2PasswordRequestForm = Depends()):
    conn = get_db_connection();
    user = conn.execute('SELECT * FROM users WHERE username=?', (fd.username,)).fetchone();
    conn.close()
    if not user or not verify_password(fd.password, user['hashed_password']): raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username/password")
    return {"access_token": create_access_token(data={"sub": user['username'], "user_id": user['id']}),
            "token_type": "bearer"}


@fastapi_app.get("/users/me", response_model=UserResponse, dependencies=[Depends(rate_limiter_dependency)])
async def read_users_me_endpoint(cu: dict = Depends(get_current_active_user)):
    reset_user_credits_if_needed(cu['id']);
    conn = get_db_connection();
    udr = conn.execute('SELECT * FROM users WHERE id=?', (cu['id'],)).fetchone();
    conn.close()

    user_data_dict = dict(udr)
    if user_data_dict.get('avatar_filename'):
        user_data_dict['avatar_url'] = f"/user_uploads/avatars/{user_data_dict['avatar_filename']}"
    else:
        user_data_dict['avatar_url'] = None

    return UserResponse.model_validate(user_data_dict)


@fastapi_app.post("/users/me/avatar", response_model=UserResponse, dependencies=[Depends(rate_limiter_dependency)])
async def upload_avatar_endpoint(file: UploadFile = File(...),
                                 cu: dict = Depends(get_current_active_user)):
    if not file.content_type or not file.content_type.startswith("image/"): raise HTTPException(status_code=400,
                                                                                                detail="Invalid file type.")
    ext = os.path.splitext(file.filename)[1] if file.filename else ".png";
    fn = f"avatar_user_{cu['id']}_{uuid.uuid4().hex}{ext}";
    fp = os.path.join(AVATARS_DIR, fn)
    conn = get_db_connection();
    old = conn.execute('SELECT avatar_filename FROM users WHERE id=?', (cu['id'],)).fetchone()
    if old and old['avatar_filename']: old_fp = os.path.join(AVATARS_DIR, old['avatar_filename']);os.remove(
        old_fp) if os.path.exists(old_fp) else None
    try:
        with open(fp, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        conn.close();
        raise HTTPException(status_code=500, detail=f"Avatar save error: {e}")
    finally:
        file.file.close() if file.file else None
    conn.execute('UPDATE users SET avatar_filename=? WHERE id=?', (fn, cu['id']));
    conn.commit();
    upd_user = conn.execute('SELECT * FROM users WHERE id=?', (cu['id'],)).fetchone();
    conn.close()

    user_data_dict = dict(upd_user)
    if user_data_dict.get('avatar_filename'):
        user_data_dict['avatar_url'] = f"/user_uploads/avatars/{user_data_dict['avatar_filename']}"
    else:
        user_data_dict['avatar_url'] = None

    return UserResponse.model_validate(user_data_dict)


@fastapi_app.post("/request-password-reset")
async def req_pwd_reset(req: PasswordResetRequest, bg: BackgroundTasks):
    conn = get_db_connection();
    user = conn.execute("SELECT id,email FROM users WHERE email=?", (req.email,)).fetchone()
    if user: token = str(uuid.uuid4());exp = datetime.now(timezone.utc) + timedelta(hours=1);conn.execute(
        "INSERT INTO password_reset_tokens (user_id,token,expires_at) VALUES (?,?,?)", (user['id'], token,
                                                                                        exp.isoformat()));conn.commit();link = f"{BASE_URL}/reset-with-token?token={token}";bg.add_task(
        send_email_stub, user['email'], "Password Reset", f"Token: {token} or Link: {link}")
    conn.close();
    return {"message": "If email exists, reset instructions sent."}


@fastapi_app.post("/confirm-password-reset")
async def conf_pwd_reset(req: PasswordResetConfirm):
    conn = get_db_connection();
    td = conn.execute("SELECT * FROM password_reset_tokens WHERE token=? AND used=FALSE AND expires_at > ?",
                      (req.token, datetime.now(timezone.utc).isoformat())).fetchone()
    if not td: conn.close();raise HTTPException(status_code=400, detail="Invalid/expired token.")
    hp = get_password_hash(req.new_password);
    conn.execute("UPDATE users SET hashed_password=? WHERE id=?", (hp, td['user_id']));
    conn.execute("UPDATE password_reset_tokens SET used=TRUE WHERE id=?", (td['id'],));
    conn.commit();
    conn.close()
    return {"message": "Password reset successfully."}


@fastapi_app.get("/subscriptions/plans", dependencies=[Depends(rate_limiter_dependency)])
async def get_plans_packs(): return {
    id: {"name": d["name"], "type": d["type"], "price": d.get("price_monthly", d.get("price")),
         "image_credits": d.get("image_credits_monthly", d.get("credits_awarded")),
         "max_resolution": d.get("max_resolution", "N/A"), "features": d["features"]} for id, d in
    SUBSCRIPTION_PLANS.items()}


@fastapi_app.post("/subscriptions/create-payment-session", dependencies=[Depends(rate_limiter_dependency)])
async def create_pay_sess(pr: PaymentRequest, cu: dict = Depends(get_current_active_user)):
    if pr.item_id not in SUBSCRIPTION_PLANS: raise HTTPException(status_code=400, detail="Invalid item.")
    if pr.payment_provider == 'stripe': return await create_stripe_checkout_session(cu['id'], pr.item_id,
                                                                                    pr.coupon_code)
    raise HTTPException(status_code=400, detail="Unsupported payment provider.")


@fastapi_app.get("/subscriptions/manage-portal", dependencies=[Depends(rate_limiter_dependency)])
async def get_stripe_portal(cu: dict = Depends(get_current_active_user)):
    if not STRIPE_SECRET_KEY or STRIPE_SECRET_KEY.startswith("sk_test_YOUR") or STRIPE_CUSTOMER_PORTAL_LINK.startswith(
            "YOUR"): raise HTTPException(status_code=503, detail="Stripe/Portal not configured.")
    if not cu['stripe_customer_id']: raise HTTPException(status_code=404, detail="No Stripe customer ID.")
    import stripe;
    stripe.api_key = STRIPE_SECRET_KEY
    try:
        pcfgs = stripe.billing_portal.Configuration.list(limit=1, active=True);
    except Exception as e:
        print(f"Could not list portal configs: {e}")
    try:
        sess = stripe.billing_portal.Session.create(customer=cu['stripe_customer_id'],
                                                    return_url=f"{BASE_URL}/payment-success");
        return {
            "portal_url": sess.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stripe Portal Error: {e}")


@fastapi_app.get("/users/me/billing-history", response_model=list[dict],
                 dependencies=[Depends(rate_limiter_dependency)])
async def get_billing(cu: dict = Depends(get_current_active_user)):
    conn = get_db_connection();
    pays = conn.execute(
        "SELECT item_id,amount,currency,status,payment_gateway,transaction_id,created_at FROM payments WHERE user_id=? ORDER BY created_at DESC",
        (cu['id'],)).fetchall();
    conn.close()
    return [{**dict(p), "item_name": SUBSCRIPTION_PLANS.get(p["item_id"], {}).get("name", p["item_id"])} for p in pays]


@fastapi_app.post("/upload/img2img-base", dependencies=[Depends(rate_limiter_dependency)])
async def upload_i2i_base(file: UploadFile = File(...), cu: dict = Depends(get_current_active_user)):
    if not file.content_type or not file.content_type.startswith("image/"): raise HTTPException(status_code=400,
                                                                                                detail="Invalid file type.")
    ext = os.path.splitext(file.filename)[1] if file.filename else ".png";
    fn = f"baseimg_user_{cu['id']}_{uuid.uuid4().hex}{ext}";
    fp = os.path.join(IMG2IMG_BASE_DIR, fn)
    try:
        with open(fp, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Base image save error: {e}")
    finally:
        file.file.close() if file.file else None
    return {"filename": fn, "message": "Base image uploaded."}


@fastapi_app.post("/generate/image", dependencies=[Depends(rate_limiter_dependency)])
async def api_generate_image_v3(payload: PromptPayload, current_user: dict = Depends(get_current_active_user)):
    user_id = current_user['id']
    results, errors = [], []
    for i in range(payload.num_images):
        seed = payload.seed + i if payload.seed != 0 else 0
        base_path = None
        if payload.base_image_filename:
            path_candidate = os.path.join(IMG2IMG_BASE_DIR, payload.base_image_filename)
            if os.path.exists(path_candidate) and os.path.commonpath([IMG2IMG_BASE_DIR]) == os.path.commonpath(
                    [IMG2IMG_BASE_DIR, path_candidate]):
                base_path = path_candidate
            else:
                errors.append(f"Base image '{payload.base_image_filename}' invalid for image {i + 1}.");
                continue

        try:
            engine = payload.engine_id.lower()
            if "sdxl" in engine or engine == "stable-diffusion-xl-1024-v1-0":
                res = await generate_image_sdxl(payload.prompt, user_id, payload.negative_prompt, payload.width,
                                                payload.height, payload.steps, payload.cfg_scale, seed, base_path,
                                                payload.image_strength)
            elif engine == "dall-e-3":
                res = await generate_image_dalle3(payload.prompt, user_id, payload.width, payload.height)
            elif engine.startswith("stable-diffusion"):
                res = await generate_image_stabilityai(payload.prompt, user_id, payload.negative_prompt,
                                                       payload.engine_id, payload.steps, payload.cfg_scale,
                                                       payload.width, payload.height, seed, base_path,
                                                       payload.image_strength)
            else:
                errors.append(f"Unsupported engine: {payload.engine_id} for image {i + 1}.");
                continue
            results.append(res)
        except HTTPException as h_exc:
            errors.append(f"Img {i + 1} ({payload.engine_id}): {h_exc.detail}")
        except Exception as exc:
            errors.append(f"Img {i + 1} ({payload.engine_id}) unexpected error: {exc}")

    msg = "Generation completed." + (f" Errors: {'; '.join(errors)}" if errors else "")
    if not results and errors: raise HTTPException(status_code=500,
                                                   detail=f"All generations failed. Errors: {'; '.join(errors)}")
    return {"generated_images": results, "message": msg, "errors": errors}


@fastapi_app.get("/gallery/images", dependencies=[Depends(rate_limiter_dependency)])
async def list_user_images_v3(cu: dict = Depends(get_current_active_user), page: int = Query(default=1, ge=1),
                              page_size: int = Query(default=12, ge=1, le=48),
                              sort_by: str = Query(default="generated_at"),
                              sort_order: str = Query(default="desc"), filter_prompt: str | None = Query(default=None),
                              filter_tags: str | None = Query(default=None)):
    conn = get_db_connection();
    offset = (page - 1) * page_size;
    base_q = 'FROM image_generations WHERE user_id=?';
    base_p = [cu['id']]
    filters = [];
    if filter_prompt: filters.append('prompt LIKE ?');base_p.append(f'%{filter_prompt}%')
    if filter_tags:
        for tag in [t.strip() for t in filter_tags.split(',') if t.strip()]: filters.append(
            'tags LIKE ?');base_p.append(f'%{tag}%')
    if filters: base_q += f' AND {" AND ".join(filters)}'
    total = conn.execute(f'SELECT COUNT(*) {base_q}', tuple(base_p)).fetchone()[0]
    sort_col = sort_by if sort_by in {"generated_at", "rating", "prompt", "engine_id", "width",
                                      "height"} else "generated_at"
    sort_ord = "DESC" if sort_order.lower() == "desc" else "ASC"
    items_q = f'SELECT * {base_q} ORDER BY {sort_col} {sort_ord} LIMIT ? OFFSET ?';
    base_p.extend([page_size, offset])
    imgs = conn.execute(items_q, tuple(base_p)).fetchall();
    conn.close()
    return {"items": [dict(i) for i in imgs], "total": total, "page": page, "page_size": page_size,
            "num_pages": (total + page_size - 1) // page_size if total > 0 else 0}


@fastapi_app.put("/gallery/image/{image_id}", dependencies=[Depends(rate_limiter_dependency)])
async def update_img_details(img_id: int, data: ImageUpdateRequest,
                             cu: dict = Depends(get_current_active_user)):
    conn = get_db_connection();
    img = conn.execute("SELECT id FROM image_generations WHERE id=? AND user_id=?", (img_id, cu['id'])).fetchone()
    if not img: conn.close();raise HTTPException(status_code=404, detail="Image not found/access denied.")
    updates, params = [], []
    if data.is_public is not None: updates.append("is_public=?");params.append(data.is_public)
    if data.tags is not None: updates.append("tags=?");params.append(data.tags)
    if data.rating is not None: updates.append("rating=?");params.append(data.rating)
    if not updates: conn.close();return {"message": "No update data."}
    conn.execute(f"UPDATE image_generations SET {', '.join(updates)} WHERE id=?", tuple(params + [img_id]));
    conn.commit()
    upd_img = conn.execute("SELECT * FROM image_generations WHERE id=?", (img_id,)).fetchone();
    conn.close()
    return dict(upd_img)


@fastapi_app.delete("/gallery/image/{image_id}", status_code=status.HTTP_204_NO_CONTENT,
                    dependencies=[Depends(rate_limiter_dependency)])
async def del_img_gallery(img_id: int, cu: dict = Depends(get_current_active_user)):
    conn = get_db_connection();
    img = conn.execute("SELECT image_filename FROM image_generations WHERE id=? AND user_id=?",
                       (img_id, cu['id'])).fetchone()
    if not img: conn.close();raise HTTPException(status_code=404, detail="Image not found/access denied.")
    if img['image_filename']: fp = os.path.join(GENERATED_IMAGES_DIR, img['image_filename']);os.remove(
        fp) if os.path.exists(fp) else None
    conn.execute("DELETE FROM image_generations WHERE id=?", (img_id,));
    conn.commit();
    conn.close()


# --- TKINTER FRONTEND (CustomTkinter) with UI/UX Enhancements ---
class AdvancedUltimateAIGeneratorApp(ctk.CTk):
    def __init__(self, fastapi_url="http://localhost:8000"):
        super().__init__()
        self.fastapi_url = fastapi_url
        self.access_token = None;
        self.user_info = None;
        self.generated_image_path = None
        self.displayed_image = None  # Store the PIL Image object of the currently displayed image
        self.image_gallery_items = [];
        self.current_base_image_for_img2img_local_path = None
        self.current_base_image_filename_server = None
        self.icons = {}

        self._apply_initial_styling()
        self.title("GenMaster AI Studio ✨")
        self.geometry("1400x950")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._load_icons()
        self._create_all_frames_and_widgets()

        self.load_token()
        if self.access_token:
            self.fetch_user_info_async()
        else:
            self.show_frame("login");
            if hasattr(self, 'nav_frame'): self.nav_frame.grid_remove()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _apply_initial_styling(self):
        ctk.set_appearance_mode("Dark" if STYLE_CONFIG["dark_mode"] else "Light")
        ctk.set_default_color_theme(STYLE_CONFIG["theme_color"])
        self.configure(fg_color=get_style("APP_BG_COLOR"))

    def _load_icons(self):
        icon_files = {
            "generate": "palette_FILL0_wght400_GRAD0_opsz24.png",
            "gallery": "photo_library_FILL0_wght400_GRAD0_opsz24.png",
            "subscription": "workspace_premium_FILL0_wght400_GRAD0_opsz24.png",
            "profile": "account_circle_FILL0_wght400_GRAD0_opsz24.png",
            "logout": "logout_FILL0_wght400_GRAD0_opsz24.png",
            "upload": "upload_file_FILL0_wght400_GRAD0_opsz24.png",
            "save": "save_FILL0_wght400_GRAD0_opsz24.png",
            "delete": "delete_FILL0_wght400_GRAD0_opsz24.png",
            "edit": "edit_FILL0_wght400_GRAD0_opsz24.png"
        }
        for name, filename in icon_files.items():
            try:
                icon_path = os.path.join(ICONS_DIR, filename)
                if os.path.exists(icon_path):
                    img = Image.open(icon_path).convert("RGBA")
                    img = img.resize((20, 20), Image.LANCZOS)
                    self.icons[name] = ctk.CTkImage(light_image=img, dark_image=img, size=(20, 20))
                else:
                    self.icons[name] = None
            except Exception as e:
                print(f"Error loading icon {name}: {e}")
                self.icons[name] = None
        for key in icon_files.keys():
            if key not in self.icons: self.icons[key] = None

    def _create_styled_button(self, master, text, command, icon_name=None, **kwargs):
        btn_kwargs = {
            "corner_radius": STYLE_CONFIG["CORNER_RADIUS"],
            "font": (
                STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"], STYLE_CONFIG["FONT_WEIGHT_BOLD"]),
            "fg_color": get_style("BUTTON_COLOR"),
            "hover_color": get_style("BUTTON_HOVER_COLOR"),
            "text_color": get_style("BUTTON_TEXT_COLOR"),  # Use specific button text color
            "height": 36,
            "image": self.icons.get(icon_name) if icon_name and self.icons.get(icon_name) else None,
            "compound": "left" if icon_name and self.icons.get(icon_name) else "none",
            "anchor": "center"
        }
        btn_kwargs.update(kwargs)
        return ctk.CTkButton(master, text=text, command=command, **btn_kwargs)

    def _create_styled_frame(self, master, **kwargs):
        frame_kwargs = {
            "fg_color": get_style("FRAME_BG_COLOR"),
            "corner_radius": STYLE_CONFIG["CORNER_RADIUS"],
            "border_width": 1,
            "border_color": get_style("BORDER_COLOR")
        }
        frame_kwargs.update(kwargs)
        return ctk.CTkFrame(master, **frame_kwargs)

    def _create_section_label(self, master, text):
        return ctk.CTkLabel(master, text=text,
                            font=(STYLE_CONFIG["FONT_FAMILY_HEADINGS"], STYLE_CONFIG["FONT_SIZE_LARGE"],
                                  STYLE_CONFIG["FONT_WEIGHT_BOLD"]),
                            text_color=get_style("TEXT_COLOR"), anchor="w")

    def _create_all_frames_and_widgets(self):
        self.nav_frame = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color=get_style("FRAME_BG_COLOR"),
                                      border_width=0)
        self.nav_frame.grid(row=0, column=0, sticky="nsw")
        self.nav_frame.grid_rowconfigure(5, weight=1)

        self.logo_label = ctk.CTkLabel(self.nav_frame, text="GenMaster AI",
                                       font=(STYLE_CONFIG["FONT_FAMILY_HEADINGS"], STYLE_CONFIG["FONT_SIZE_XLARGE"],
                                             STYLE_CONFIG["FONT_WEIGHT_BOLD"]),
                                       text_color=get_style("ACCENT_COLOR"))
        self.logo_label.grid(row=0, column=0, padx=STYLE_CONFIG["MAIN_PADX"],
                             pady=(STYLE_CONFIG["MAIN_PADY"], STYLE_CONFIG["SECTION_PADY"]))

        nav_buttons_data = [
            ("Generate", "generate", "generate"), ("Gallery", "gallery", "gallery"),
            ("Subscription", "subscription", "subscription"), ("My Profile", "profile", "profile")
        ]
        for i, (text, frame_name, icon_name) in enumerate(nav_buttons_data):
            # FIX: Use ACCENT_COLOR for these transparent buttons for better visibility and style.
            btn = self._create_styled_button(self.nav_frame, text, lambda fn=frame_name: self.show_frame(fn),
                                             icon_name=icon_name,
                                             fg_color="transparent", hover_color=get_style("INPUT_BG_COLOR"),
                                             text_color=get_style("ACCENT_COLOR"), anchor="w", width=200)
            btn.grid(row=i + 1, column=0, padx=STYLE_CONFIG["WIDGET_PADX"], pady=STYLE_CONFIG["WIDGET_PADY"] // 2,
                     sticky="ew")

        self.appearance_mode_menu = ctk.CTkOptionMenu(self.nav_frame, values=["Light", "Dark", "System"],
                                                      command=self.change_appearance_mode_event,
                                                      font=(STYLE_CONFIG["FONT_FAMILY_MAIN"],
                                                            STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                                                      fg_color=get_style("INPUT_BG_COLOR"),
                                                      button_color=get_style("BUTTON_COLOR"),
                                                      button_hover_color=get_style("BUTTON_HOVER_COLOR"),
                                                      dropdown_fg_color=get_style("INPUT_BG_COLOR"),
                                                      text_color=get_style("TEXT_COLOR"))
        self.appearance_mode_menu.grid(row=len(nav_buttons_data) + 1, column=0, padx=STYLE_CONFIG["WIDGET_PADX"],
                                       pady=STYLE_CONFIG["WIDGET_PADY"], sticky="ew")
        self.appearance_mode_menu.set("Dark" if STYLE_CONFIG["dark_mode"] else "Light")

        self.logout_button_nav = self._create_styled_button(self.nav_frame, "Logout", self.logout, icon_name="logout",
                                                            fg_color="transparent", border_width=1,
                                                            border_color=get_style("BORDER_COLOR"),
                                                            text_color=get_style("TEXT_DISABLED_COLOR"), anchor="w",
                                                            width=200)
        self.logout_button_nav.grid(row=len(nav_buttons_data) + 2, column=0, padx=STYLE_CONFIG["WIDGET_PADX"],
                                    pady=(STYLE_CONFIG["WIDGET_PADY"], STYLE_CONFIG["MAIN_PADY"]), sticky="sew")

        self.main_area = ctk.CTkFrame(self, fg_color="transparent")
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=STYLE_CONFIG["MAIN_PADX"],
                            pady=STYLE_CONFIG["MAIN_PADY"])
        self.main_area.grid_columnconfigure(0, weight=1);
        self.main_area.grid_rowconfigure(0, weight=1)

        self.frames = {}
        frame_names = ["login", "register", "generate", "subscription", "gallery", "profile",
                       "password_reset_request", "password_reset_confirm"]
        for F_name in frame_names:
            frame = ctk.CTkFrame(self.main_area, fg_color="transparent")
            self.frames[F_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self._create_login_frame();
        self._create_register_frame();
        self._create_generate_frame()
        self._create_subscription_frame();
        self._create_gallery_frame();
        self._create_profile_frame()
        self._create_password_reset_frames()

        self.status_bar = ctk.CTkLabel(self, text="Welcome to GenMaster AI Studio!", anchor="w", height=30,
                                       font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                                       text_color=get_style("TEXT_COLOR"), fg_color=get_style("FRAME_BG_COLOR"))
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=0, pady=0)

    def _create_login_frame(self):
        frame = self.frames["login"]
        frame.grid_columnconfigure(0, weight=1);
        frame.grid_rowconfigure(0, weight=1)

        login_form_container = self._create_styled_frame(frame, width=400, height=450)
        login_form_container.grid(row=0, column=0, sticky="", padx=STYLE_CONFIG["MAIN_PADX"],
                                  pady=STYLE_CONFIG["MAIN_PADY"])
        login_form_container.grid_propagate(False)
        login_form_container.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(login_form_container, text="Welcome Back!",
                     font=(STYLE_CONFIG["FONT_FAMILY_HEADINGS"], STYLE_CONFIG["FONT_SIZE_XLARGE"] + 2,
                           STYLE_CONFIG["FONT_WEIGHT_BOLD"]),
                     text_color=get_style("TEXT_COLOR")).pack(
            pady=(STYLE_CONFIG["MAIN_PADY"] + 10, STYLE_CONFIG["SECTION_PADY"]))
        ctk.CTkLabel(login_form_container, text="Login to continue your creative journey.",
                     font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                     text_color=get_style("TEXT_DISABLED_COLOR")).pack(pady=(0, STYLE_CONFIG["SECTION_PADY"]))

        username_entry = ctk.CTkEntry(login_form_container, placeholder_text="Username", width=300, height=40,
                                      corner_radius=STYLE_CONFIG["CORNER_RADIUS"],
                                      font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                                      text_color=get_style("TEXT_COLOR"), fg_color=get_style("INPUT_BG_COLOR"),
                                      border_color=get_style("BORDER_COLOR"))
        username_entry.pack(pady=STYLE_CONFIG["WIDGET_PADY"])
        self.login_username_entry = username_entry

        password_entry = ctk.CTkEntry(login_form_container, placeholder_text="Password", show="*", width=300, height=40,
                                      corner_radius=STYLE_CONFIG["CORNER_RADIUS"],
                                      font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                                      text_color=get_style("TEXT_COLOR"), fg_color=get_style("INPUT_BG_COLOR"),
                                      border_color=get_style("BORDER_COLOR"))
        password_entry.pack(pady=STYLE_CONFIG["WIDGET_PADY"])
        self.login_password_entry = password_entry
        password_entry.bind("<Return>", lambda event: self.login())

        self._create_styled_button(login_form_container, text="Login", command=self.login, width=300).pack(
            pady=STYLE_CONFIG["SECTION_PADY"])

        links_frame = ctk.CTkFrame(login_form_container, fg_color="transparent")
        links_frame.pack(fill="x", pady=(0, STYLE_CONFIG["WIDGET_PADY"]))
        self._create_styled_button(links_frame, text="Register Account", command=lambda: self.show_frame("register"),
                                   fg_color="transparent", text_color=get_style("ACCENT_COLOR"), hover=False).pack(
            side="left", expand=True)
        self._create_styled_button(links_frame, text="Forgot Password?",
                                   command=lambda: self.show_frame("password_reset_request"), fg_color="transparent",
                                   text_color=get_style("ACCENT_COLOR"), hover=False).pack(side="right", expand=True)

        self.login_status_label = ctk.CTkLabel(login_form_container, text="", wraplength=300, font=(
            STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                                               text_color=get_style("TEXT_COLOR"))
        self.login_status_label.pack(pady=STYLE_CONFIG["WIDGET_PADY"])

    def _create_register_frame(self):
        frame = self.frames["register"]
        frame.grid_columnconfigure(0, weight=1);
        frame.grid_rowconfigure(0, weight=1)

        form_container = self._create_styled_frame(frame, width=400, height=500)
        form_container.grid(row=0, column=0, sticky="", padx=STYLE_CONFIG["MAIN_PADX"], pady=STYLE_CONFIG["MAIN_PADY"])
        form_container.grid_propagate(False);
        form_container.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(form_container, text="Join GenMaster AI", font=(
            STYLE_CONFIG["FONT_FAMILY_HEADINGS"], STYLE_CONFIG["FONT_SIZE_XLARGE"] + 2,
            STYLE_CONFIG["FONT_WEIGHT_BOLD"]),
                     text_color=get_style("TEXT_COLOR")).pack(
            pady=(STYLE_CONFIG["MAIN_PADY"] + 10, STYLE_CONFIG["SECTION_PADY"]))

        self.register_username_entry = ctk.CTkEntry(form_container, placeholder_text="Username", width=300, height=40,
                                                    corner_radius=STYLE_CONFIG["CORNER_RADIUS"], font=(
                STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]), text_color=get_style("TEXT_COLOR"),
                                                    fg_color=get_style("INPUT_BG_COLOR"),
                                                    border_color=get_style("BORDER_COLOR"))
        self.register_username_entry.pack(pady=STYLE_CONFIG["WIDGET_PADY"])
        self.register_email_entry = ctk.CTkEntry(form_container, placeholder_text="Email Address", width=300, height=40,
                                                 corner_radius=STYLE_CONFIG["CORNER_RADIUS"], font=(
                STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]), text_color=get_style("TEXT_COLOR"),
                                                 fg_color=get_style("INPUT_BG_COLOR"),
                                                 border_color=get_style("BORDER_COLOR"))
        self.register_email_entry.pack(pady=STYLE_CONFIG["WIDGET_PADY"])
        self.register_password_entry = ctk.CTkEntry(form_container, placeholder_text="Password", show="*", width=300,
                                                    height=40, corner_radius=STYLE_CONFIG["CORNER_RADIUS"], font=(
                STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]), text_color=get_style("TEXT_COLOR"),
                                                    fg_color=get_style("INPUT_BG_COLOR"),
                                                    border_color=get_style("BORDER_COLOR"))
        self.register_password_entry.pack(pady=STYLE_CONFIG["WIDGET_PADY"])
        self.register_password_entry.bind("<Return>", lambda event: self.register())

        self._create_styled_button(form_container, text="Create Account", command=self.register, width=300).pack(
            pady=STYLE_CONFIG["SECTION_PADY"])
        self._create_styled_button(form_container, text="Already have an account? Login",
                                   command=lambda: self.show_frame("login"), fg_color="transparent",
                                   text_color=get_style("ACCENT_COLOR"), hover=False).pack()

        self.register_status_label = ctk.CTkLabel(form_container, text="", wraplength=300, font=(
            STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]), text_color=get_style("TEXT_COLOR"))
        self.register_status_label.pack(pady=STYLE_CONFIG["WIDGET_PADY"])

    def _create_password_reset_frames(self):
        req_frame = self.frames["password_reset_request"]
        req_frame.grid_columnconfigure(0, weight=1);
        req_frame.grid_rowconfigure(0, weight=1)
        form_container1 = self._create_styled_frame(req_frame, width=400, height=350);
        form_container1.grid(sticky="", pady=STYLE_CONFIG["MAIN_PADY"])
        form_container1.grid_propagate(False);
        form_container1.grid_columnconfigure(0, weight=1)

        self._create_section_label(form_container1, "Reset Password").pack(
            pady=(STYLE_CONFIG["MAIN_PADY"], STYLE_CONFIG["SECTION_PADY"]))
        ctk.CTkLabel(form_container1, text="Enter your email to receive a reset link.",
                     font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                     text_color=get_style("TEXT_DISABLED_COLOR")).pack(pady=(0, STYLE_CONFIG["WIDGET_PADY"]))
        self.reset_email_entry = ctk.CTkEntry(form_container1, placeholder_text="Email Address", width=300, height=40,
                                              corner_radius=STYLE_CONFIG["CORNER_RADIUS"],
                                              font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                                              text_color=get_style("TEXT_COLOR"), fg_color=get_style("INPUT_BG_COLOR"),
                                              border_color=get_style("BORDER_COLOR"))
        self.reset_email_entry.pack(pady=STYLE_CONFIG["WIDGET_PADY"])
        self.reset_email_entry.bind("<Return>", lambda event: self.request_password_reset_action())
        self._create_styled_button(form_container1, "Send Reset Link", self.request_password_reset_action,
                                   width=300).pack(pady=STYLE_CONFIG["WIDGET_PADY"])
        self.reset_req_status_label = ctk.CTkLabel(form_container1, text="", wraplength=300, font=(
            STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]), text_color=get_style("TEXT_COLOR"));
        self.reset_req_status_label.pack(pady=STYLE_CONFIG["WIDGET_PADY"])
        self._create_styled_button(form_container1, "Back to Login", lambda: self.show_frame("login"),
                                   fg_color="transparent", text_color=get_style("ACCENT_COLOR"), hover=False).pack()

        conf_frame = self.frames["password_reset_confirm"]
        conf_frame.grid_columnconfigure(0, weight=1);
        conf_frame.grid_rowconfigure(0, weight=1)
        form_container2 = self._create_styled_frame(conf_frame, width=400, height=400);
        form_container2.grid(sticky="", pady=STYLE_CONFIG["MAIN_PADY"])
        form_container2.grid_propagate(False);
        form_container2.grid_columnconfigure(0, weight=1)

        self._create_section_label(form_container2, "Confirm New Password").pack(
            pady=(STYLE_CONFIG["MAIN_PADY"], STYLE_CONFIG["SECTION_PADY"]))
        self.reset_token_entry = ctk.CTkEntry(form_container2, placeholder_text="Reset Token (from email)", width=300,
                                              height=40, corner_radius=STYLE_CONFIG["CORNER_RADIUS"],
                                              font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                                              text_color=get_style("TEXT_COLOR"), fg_color=get_style("INPUT_BG_COLOR"),
                                              border_color=get_style("BORDER_COLOR"))
        self.reset_token_entry.pack(pady=STYLE_CONFIG["WIDGET_PADY"])
        self.reset_new_password_entry = ctk.CTkEntry(form_container2, placeholder_text="New Password", show="*",
                                                     width=300, height=40, corner_radius=STYLE_CONFIG["CORNER_RADIUS"],
                                                     font=(STYLE_CONFIG["FONT_FAMILY_MAIN"],
                                                           STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                                                     text_color=get_style("TEXT_COLOR"),
                                                     fg_color=get_style("INPUT_BG_COLOR"),
                                                     border_color=get_style("BORDER_COLOR"))
        self.reset_new_password_entry.pack(pady=STYLE_CONFIG["WIDGET_PADY"])
        self.reset_new_password_entry.bind("<Return>", lambda event: self.confirm_password_reset_action())
        self._create_styled_button(form_container2, "Set New Password", self.confirm_password_reset_action,
                                   width=300).pack(pady=STYLE_CONFIG["WIDGET_PADY"])
        self.reset_conf_status_label = ctk.CTkLabel(form_container2, text="", wraplength=300, font=(
            STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]), text_color=get_style("TEXT_COLOR"));
        self.reset_conf_status_label.pack(pady=STYLE_CONFIG["WIDGET_PADY"])

    def _create_generate_frame(self):
        frame = self.frames["generate"]
        frame.grid_columnconfigure(0, weight=2);
        frame.grid_columnconfigure(1, weight=3)
        frame.grid_rowconfigure(0, weight=1)

        input_area_container = ctk.CTkFrame(frame, fg_color="transparent")
        input_area_container.grid(row=0, column=0, padx=(0, STYLE_CONFIG["WIDGET_PADX"]), pady=0, sticky="nsew")
        input_area_container.grid_rowconfigure(0, weight=1);
        input_area_container.grid_columnconfigure(0, weight=1)

        input_scroll_area = ctk.CTkScrollableFrame(input_area_container, fg_color=get_style("FRAME_BG_COLOR"),
                                                   corner_radius=STYLE_CONFIG["CORNER_RADIUS"], border_width=1,
                                                   border_color=get_style("BORDER_COLOR"))
        input_scroll_area.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        input_scroll_area.grid_columnconfigure(0, weight=1)

        prompt_section = self._create_styled_frame(input_scroll_area, fg_color="transparent", border_width=0)
        prompt_section.pack(fill="x", padx=STYLE_CONFIG["WIDGET_PADX"], pady=STYLE_CONFIG["SECTION_PADY"])
        self._create_section_label(prompt_section, "Creative Prompting").pack(fill="x",
                                                                              pady=(0, STYLE_CONFIG["WIDGET_PADY"]))
        self.prompt_entry = ctk.CTkTextbox(prompt_section, height=100, wrap="word",
                                           corner_radius=STYLE_CONFIG["CORNER_RADIUS"],
                                           font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                                           border_width=1, border_color=get_style("BORDER_COLOR"),
                                           text_color=get_style("TEXT_COLOR"),
                                           fg_color=get_style("INPUT_BG_COLOR"))
        self.prompt_entry.pack(fill="x", pady=(0, STYLE_CONFIG["WIDGET_PADY"]))
        self.prompt_entry.insert("0.0",
                                 "Epic digital painting of a futuristic city at sunset, volumetric lighting, hyperdetailed, trending on artstation")
        self.neg_prompt_entry = ctk.CTkTextbox(prompt_section, height=50, wrap="word",
                                               corner_radius=STYLE_CONFIG["CORNER_RADIUS"], font=(
                STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]), border_width=1,
                                               border_color=get_style("BORDER_COLOR"),
                                               text_color=get_style("TEXT_COLOR"), fg_color=get_style("INPUT_BG_COLOR"))
        self.neg_prompt_entry.pack(fill="x", pady=(0, STYLE_CONFIG["WIDGET_PADY"]))
        self.neg_prompt_entry.insert("0.0",
                                     "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft")

        model_params_section = self._create_styled_frame(input_scroll_area, fg_color="transparent", border_width=0)
        model_params_section.pack(fill="x", padx=STYLE_CONFIG["WIDGET_PADX"], pady=STYLE_CONFIG["SECTION_PADY"])
        self._create_section_label(model_params_section, "Model & Parameters").pack(fill="x", pady=(
            0, STYLE_CONFIG["WIDGET_PADY"]))

        ctk.CTkLabel(model_params_section, text="AI Model Engine:",
                     font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                     text_color=get_style("TEXT_COLOR")).pack(anchor="w")
        self.engine_options = ["stable-diffusion-xl-1024-v1-0", "stable-diffusion-v1-6", "dall-e-3"]
        self.engine_id_menu = ctk.CTkOptionMenu(model_params_section, values=self.engine_options,
                                                command=self.on_engine_selected, font=(
                STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                                                corner_radius=STYLE_CONFIG["CORNER_RADIUS"],
                                                fg_color=get_style("INPUT_BG_COLOR"),
                                                button_color=get_style("BUTTON_COLOR"),
                                                button_hover_color=get_style("BUTTON_HOVER_COLOR"),
                                                dropdown_fg_color=get_style("INPUT_BG_COLOR"),
                                                text_color=get_style("TEXT_COLOR"))
        self.engine_id_menu.pack(fill="x", pady=(0, STYLE_CONFIG["WIDGET_PADY"]))

        self.param_sliders = {};
        self.param_value_labels = {}
        slider_params = [("Width:", "width", 256, 2048, 512, (2048 - 256) // 64, "px"),
                         ("Height:", "height", 256, 2048, 512, (2048 - 256) // 64, "px"),
                         ("Steps:", "steps", 10, 100, 30, 18, ""),
                         ("CFG Scale:", "cfg", 1.0, 20.0, 7.0, 190, "")]
        for label_text, key, p_from, p_to, p_default, p_steps, suffix in slider_params:
            row = ctk.CTkFrame(model_params_section, fg_color="transparent");
            row.pack(fill="x", pady=STYLE_CONFIG["WIDGET_PADY"] // 2)
            ctk.CTkLabel(row, text=label_text,
                         font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]), width=100,
                         anchor="w", text_color=get_style("TEXT_COLOR")).pack(side="left")
            self.param_sliders[key] = ctk.CTkSlider(row, from_=p_from, to=p_to, number_of_steps=p_steps,
                                                    fg_color=get_style("INPUT_BG_COLOR"),
                                                    progress_color=get_style("ACCENT_COLOR"),
                                                    button_color=get_style("BUTTON_COLOR"),
                                                    button_hover_color=get_style("BUTTON_HOVER_COLOR"));
            self.param_sliders[key].set(p_default)
            self.param_sliders[key].pack(side="left", fill="x", expand=True, padx=STYLE_CONFIG["WIDGET_PADX"])
            self.param_value_labels[key] = ctk.CTkLabel(row, text=f"{p_default}{suffix}", font=(
                STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]), width=60, anchor="e",
                                                        text_color=get_style("TEXT_COLOR"));
            self.param_value_labels[key].pack(side="left")

            cmd = (
                lambda val, k_param=key, sfx_param=suffix, lbl_param=self.param_value_labels[key]: lbl_param.configure(
                    text=f"{val:.1f}{sfx_param}" if k_param == "cfg" else f"{int(val) // (64 if 'px' in sfx_param else 1) * (64 if 'px' in sfx_param else 1)}{sfx_param}"))
            self.param_sliders[key].configure(command=cmd)

        seed_batch_frame = ctk.CTkFrame(model_params_section, fg_color="transparent");
        seed_batch_frame.pack(fill="x", pady=STYLE_CONFIG["WIDGET_PADY"])
        ctk.CTkLabel(seed_batch_frame, text="Seed:",
                     font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                     text_color=get_style("TEXT_COLOR")).pack(side="left", padx=(0, 5))
        self.seed_entry = ctk.CTkEntry(seed_batch_frame, placeholder_text="0 (Random)", width=120,
                                       corner_radius=STYLE_CONFIG["CORNER_RADIUS"],
                                       font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                                       text_color=get_style("TEXT_COLOR"), fg_color=get_style("INPUT_BG_COLOR"),
                                       border_color=get_style("BORDER_COLOR"));
        self.seed_entry.insert(0, "0");
        self.seed_entry.pack(side="left", padx=(0, 10))
        ctk.CTkLabel(seed_batch_frame, text="Batch Size:",
                     font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                     text_color=get_style("TEXT_COLOR")).pack(side="left", padx=(10, 5))
        self.num_images_entry = ctk.CTkEntry(seed_batch_frame, placeholder_text="1-4", width=60,
                                             corner_radius=STYLE_CONFIG["CORNER_RADIUS"],
                                             font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                                             text_color=get_style("TEXT_COLOR"), fg_color=get_style("INPUT_BG_COLOR"),
                                             border_color=get_style("BORDER_COLOR"));
        self.num_images_entry.insert(0, "1");
        self.num_images_entry.pack(side="left")

        img2img_section = self._create_styled_frame(input_scroll_area, fg_color="transparent", border_width=0)
        img2img_section.pack(fill="x", padx=STYLE_CONFIG["WIDGET_PADX"], pady=STYLE_CONFIG["SECTION_PADY"])
        self._create_section_label(img2img_section, "Image-to-Image (Optional)").pack(fill="x", pady=(
            0, STYLE_CONFIG["WIDGET_PADY"]))

        i2i_upload_row = ctk.CTkFrame(img2img_section, fg_color="transparent");
        i2i_upload_row.pack(fill="x", pady=(0, STYLE_CONFIG["WIDGET_PADY"]))
        # FIX: Use ACCENT_COLOR for this button text for better visibility on its light background.
        self.img2img_upload_button = self._create_styled_button(i2i_upload_row, "Upload Base Image",
                                                                self.upload_base_image_for_img2img, icon_name="upload",
                                                                fg_color=get_style("INPUT_BG_COLOR"),
                                                                text_color=get_style("ACCENT_COLOR"))
        self.img2img_upload_button.pack(side="left", padx=(0, STYLE_CONFIG["WIDGET_PADX"]))
        self.img2img_base_image_label = ctk.CTkLabel(i2i_upload_row, text="No base image.", font=(
            STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                                                     text_color=get_style("TEXT_DISABLED_COLOR"), wraplength=200)
        self.img2img_base_image_label.pack(side="left", fill="x", expand=True)

        i2i_strength_row = ctk.CTkFrame(img2img_section, fg_color="transparent");
        i2i_strength_row.pack(fill="x", pady=(0, STYLE_CONFIG["WIDGET_PADY"]))
        ctk.CTkLabel(i2i_strength_row, text="Image Strength:",
                     font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]), width=120,
                     anchor="w", text_color=get_style("TEXT_COLOR")).pack(side="left")
        self.img2img_strength_slider = ctk.CTkSlider(i2i_strength_row, from_=0.0, to=1.0, number_of_steps=20,
                                                     fg_color=get_style("INPUT_BG_COLOR"),
                                                     progress_color=get_style("ACCENT_COLOR"),
                                                     button_color=get_style("BUTTON_COLOR"),
                                                     button_hover_color=get_style("BUTTON_HOVER_COLOR"));
        self.img2img_strength_slider.set(0.35)
        self.img2img_strength_slider.pack(side="left", fill="x", expand=True, padx=STYLE_CONFIG["WIDGET_PADX"])
        self.img2img_strength_label = ctk.CTkLabel(i2i_strength_row, text="0.35", font=(
            STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]), width=40, anchor="e",
                                                   text_color=get_style("TEXT_COLOR"));
        self.img2img_strength_label.pack(side="left")
        self.img2img_strength_slider.configure(
            command=lambda val: self.img2img_strength_label.configure(text=f"{val:.2f}"))

        self.generate_button_main = self._create_styled_button(input_scroll_area, "✨ Generate Masterpiece ✨",
                                                               self.trigger_generate_image_thread, height=45, font=(
                STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_MEDIUM"] + 2,
                STYLE_CONFIG["FONT_WEIGHT_BOLD"]))
        self.generate_button_main.pack(fill="x", padx=STYLE_CONFIG["WIDGET_PADX"],
                                       pady=(STYLE_CONFIG["SECTION_PADY"], STYLE_CONFIG["WIDGET_PADY"]))
        self.progress_bar = ctk.CTkProgressBar(input_scroll_area, orientation="horizontal", mode="indeterminate",
                                               progress_color=get_style("ACCENT_COLOR"),
                                               fg_color=get_style("INPUT_BG_COLOR"))
        self.generation_status_label = ctk.CTkLabel(input_scroll_area, text="",
                                                    wraplength=input_scroll_area.winfo_width() - 40, font=(
                STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                                                    text_color=get_style("TEXT_COLOR"))
        self.generation_status_label.pack(fill="x", padx=STYLE_CONFIG["WIDGET_PADX"],
                                          pady=(0, STYLE_CONFIG["WIDGET_PADY"]))

        self.on_engine_selected(self.engine_id_menu.get())

        image_display_container = self._create_styled_frame(frame)
        image_display_container.grid(row=0, column=1, padx=(STYLE_CONFIG["WIDGET_PADX"], 0), pady=0, sticky="nsew")
        image_display_container.grid_rowconfigure(0, weight=1);
        image_display_container.grid_columnconfigure(0, weight=1)

        self.image_label = ctk.CTkLabel(image_display_container, text="Your AI masterpiece awaits creation!",
                                        text_color=get_style("TEXT_DISABLED_COLOR"),
                                        font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_LARGE"]))
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=STYLE_CONFIG["WIDGET_PADX"],
                              pady=STYLE_CONFIG["WIDGET_PADY"])

        image_actions_frame = ctk.CTkFrame(image_display_container, fg_color="transparent")
        image_actions_frame.grid(row=1, column=0, sticky="ew", padx=STYLE_CONFIG["WIDGET_PADX"],
                                 pady=(0, STYLE_CONFIG["WIDGET_PADY"]))
        image_actions_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.save_image_button = self._create_styled_button(image_actions_frame, "Save Image",
                                                            self.save_displayed_image, icon_name="save",
                                                            state="disabled")
        self.save_image_button.pack(side="left", padx=STYLE_CONFIG["WIDGET_PADX"] // 2, expand=True)

    def save_displayed_image(self):
        # Placeholder implementation: add actual image saving logic
        from tkinter import filedialog, messagebox
        import PIL.Image

        if self.displayed_image is None:
            messagebox.showwarning("No Image", "No image to save.")
            return

        filepath = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if filepath:
            try:
                self.displayed_image.save(filepath)
                messagebox.showinfo("Success", f"Image saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {e}")

    def _create_subscription_frame(self):
        frame = self.frames["subscription"]
        frame.grid_columnconfigure(0, weight=1);
        frame.grid_rowconfigure(1, weight=1)

        self._create_section_label(frame, "Subscription & Credit Packs").grid(row=0, column=0,
                                                                              padx=STYLE_CONFIG["MAIN_PADX"], pady=(
                STYLE_CONFIG["MAIN_PADY"], STYLE_CONFIG["WIDGET_PADY"]), sticky="w")

        self.current_plan_info_label = ctk.CTkLabel(frame, text="Loading your current plan...", font=(
            STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_MEDIUM"]), text_color=get_style("TEXT_COLOR"),
                                                    wraplength=frame.winfo_width() - 40)
        self.current_plan_info_label.grid(row=1, column=0, padx=STYLE_CONFIG["MAIN_PADX"],
                                          pady=(0, STYLE_CONFIG["SECTION_PADY"]), sticky="ew")

        self.plans_and_packs_frame = ctk.CTkScrollableFrame(frame, fg_color="transparent", border_width=0)
        self.plans_and_packs_frame.grid(row=2, column=0, sticky="nsew", padx=STYLE_CONFIG["MAIN_PADX"] // 2, pady=0)
        self.plans_and_packs_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="plan_col")

    def _create_gallery_frame(self):
        frame = self.frames["gallery"]
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=0);
        frame.grid_rowconfigure(1, weight=0);
        frame.grid_rowconfigure(2, weight=1);
        frame.grid_rowconfigure(3, weight=0)

        self._create_section_label(frame, "Image Gallery").grid(row=0, column=0, padx=STYLE_CONFIG["MAIN_PADX"], pady=(
            STYLE_CONFIG["MAIN_PADY"], STYLE_CONFIG["WIDGET_PADY"]), sticky="w")

        controls_frame = ctk.CTkFrame(frame, fg_color="transparent")
        controls_frame.grid(row=1, column=0, pady=STYLE_CONFIG["WIDGET_PADY"], padx=STYLE_CONFIG["MAIN_PADX"],
                            sticky="ew")
        ctk.CTkLabel(controls_frame, text="Filter:",
                     font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                     text_color=get_style("TEXT_COLOR")).pack(side="left", padx=(0, 5))
        self.gallery_filter_prompt_entry = ctk.CTkEntry(controls_frame, placeholder_text="Search prompts...", width=200,
                                                        corner_radius=STYLE_CONFIG["CORNER_RADIUS"], font=(
                STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]), text_color=get_style("TEXT_COLOR"),
                                                        fg_color=get_style("INPUT_BG_COLOR"),
                                                        border_color=get_style("BORDER_COLOR"))
        self.gallery_filter_prompt_entry.pack(side="left", padx=(0, 10))
        self.gallery_filter_tags_entry = ctk.CTkEntry(controls_frame, placeholder_text="Search tags (comma-sep)...",
                                                      width=200, corner_radius=STYLE_CONFIG["CORNER_RADIUS"], font=(
                STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]), text_color=get_style("TEXT_COLOR"),
                                                      fg_color=get_style("INPUT_BG_COLOR"),
                                                      border_color=get_style("BORDER_COLOR"))
        self.gallery_filter_tags_entry.pack(side="left", padx=(0, 10))
        self._create_styled_button(controls_frame, "Apply", self.apply_gallery_filters, width=80, height=30).pack(
            side="left", padx=(0, 20))

        ctk.CTkLabel(controls_frame, text="Sort By:",
                     font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                     text_color=get_style("TEXT_COLOR")).pack(side="left", padx=(0, 5))
        self.gallery_sort_by_var = ctk.StringVar(value="generated_at")
        ctk.CTkOptionMenu(controls_frame, variable=self.gallery_sort_by_var,
                          values=["generated_at", "rating", "prompt", "engine_id"], width=130,
                          command=lambda x: self.apply_gallery_filters(),
                          font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                          corner_radius=STYLE_CONFIG["CORNER_RADIUS"], fg_color=get_style("INPUT_BG_COLOR"),
                          button_color=get_style("BUTTON_COLOR"),
                          button_hover_color=get_style("BUTTON_HOVER_COLOR"), text_color=get_style("TEXT_COLOR")).pack(
            side="left", padx=(0, 5))
        self.gallery_sort_order_var = ctk.StringVar(value="desc")
        ctk.CTkOptionMenu(controls_frame, variable=self.gallery_sort_order_var, values=["desc", "asc"], width=90,
                          command=lambda x: self.apply_gallery_filters(),
                          font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                          corner_radius=STYLE_CONFIG["CORNER_RADIUS"], fg_color=get_style("INPUT_BG_COLOR"),
                          button_color=get_style("BUTTON_COLOR"),
                          button_hover_color=get_style("BUTTON_HOVER_COLOR"), text_color=get_style("TEXT_COLOR")).pack(
            side="left")

        self.gallery_scroll_frame = ctk.CTkScrollableFrame(frame, fg_color=get_style("APP_BG_COLOR"))
        self.gallery_scroll_frame.grid(row=2, column=0, sticky="nsew", padx=STYLE_CONFIG["MAIN_PADX"] // 2, pady=0)
        self.gallery_scroll_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1, uniform="gallery_col")

        self.gallery_pagination_frame = ctk.CTkFrame(frame, fg_color="transparent")
        self.gallery_pagination_frame.grid(row=3, column=0, pady=STYLE_CONFIG["WIDGET_PADY"],
                                           padx=STYLE_CONFIG["MAIN_PADX"], sticky="ew")
        self.gallery_prev_button = self._create_styled_button(self.gallery_pagination_frame, "< Prev",
                                                              lambda: self.change_gallery_page(-1), width=100)
        self.gallery_prev_button.pack(side="left")
        self.gallery_page_label = ctk.CTkLabel(self.gallery_pagination_frame, text="Page 1/1", font=(
            STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]), text_color=get_style("TEXT_COLOR"))
        self.gallery_page_label.pack(side="left", expand=True)
        self.gallery_next_button = self._create_styled_button(self.gallery_pagination_frame, "Next >",
                                                              lambda: self.change_gallery_page(1), width=100)
        self.gallery_next_button.pack(side="right")
        self.current_gallery_page = 1;
        self.total_gallery_pages = 1

    def _create_profile_frame(self):
        frame = self.frames["profile"]
        frame.grid_columnconfigure(0, weight=1);
        frame.grid_rowconfigure(0, weight=1)

        scroll_container = ctk.CTkScrollableFrame(frame, fg_color="transparent")
        scroll_container.grid(row=0, column=0, sticky="nsew", padx=STYLE_CONFIG["MAIN_PADX"],
                              pady=STYLE_CONFIG["MAIN_PADY"])
        scroll_container.grid_columnconfigure(0, weight=1)

        content_frame = self._create_styled_frame(scroll_container)
        content_frame.pack(expand=True, fill="both", padx=STYLE_CONFIG["WIDGET_PADX"], pady=STYLE_CONFIG["WIDGET_PADY"])
        content_frame.grid_columnconfigure(1, weight=1)

        self._create_section_label(content_frame, "My Profile").grid(row=0, column=0, columnspan=2, pady=(
            STYLE_CONFIG["WIDGET_PADY"], STYLE_CONFIG["SECTION_PADY"]), sticky="w", padx=STYLE_CONFIG["WIDGET_PADX"])

        avatar_display_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        avatar_display_frame.grid(row=1, column=0, columnspan=2, pady=STYLE_CONFIG["WIDGET_PADY"], sticky="w",
                                  padx=STYLE_CONFIG["WIDGET_PADX"])
        self.profile_avatar_image_label = ctk.CTkLabel(avatar_display_frame, text="", width=100, height=100,
                                                       fg_color=get_style("INPUT_BG_COLOR"),
                                                       corner_radius=STYLE_CONFIG["CORNER_RADIUS"] * 2)
        self.profile_avatar_image_label.pack(side="left", padx=(0, STYLE_CONFIG["WIDGET_PADX"]))
        self._create_styled_button(avatar_display_frame, "Change Avatar", self.select_avatar_file,
                                   icon_name="upload").pack(side="left", anchor="center")

        info_fields = [
            ("Username:", "profile_username_label"), ("Email:", "profile_email_label"),
            ("Plan:", "profile_plan_label"), ("Credits:", "profile_credits_label"),
            ("Renews/Expires:", "profile_expiry_label"), ("Email Verified:", "profile_email_verified_label")
        ]
        for i, (text, attr_name) in enumerate(info_fields):
            ctk.CTkLabel(content_frame, text=text, font=(
                STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_MEDIUM"], STYLE_CONFIG["FONT_WEIGHT_BOLD"]),
                         text_color=get_style("TEXT_COLOR"), anchor="e").grid(row=i + 2, column=0, sticky="e",
                                                                              padx=STYLE_CONFIG["WIDGET_PADX"],
                                                                              pady=STYLE_CONFIG["WIDGET_PADY"] // 2)
            label_widget = ctk.CTkLabel(content_frame, text="N/A",
                                        font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_MEDIUM"]),
                                        text_color=get_style("TEXT_COLOR"), anchor="w")
            label_widget.grid(row=i + 2, column=1, sticky="ew", padx=STYLE_CONFIG["WIDGET_PADX"],
                              pady=STYLE_CONFIG["WIDGET_PADY"] // 2)
            setattr(self, attr_name, label_widget)

        actions_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        actions_frame.grid(row=len(info_fields) + 2, column=0, columnspan=2, pady=STYLE_CONFIG["SECTION_PADY"],
                           sticky="ew", padx=STYLE_CONFIG["WIDGET_PADX"])
        actions_frame.grid_columnconfigure(0, weight=1)

        self._create_styled_button(actions_frame, "Change Password",
                                   lambda: self.show_frame("password_reset_request")).pack(
            pady=STYLE_CONFIG["WIDGET_PADY"] // 2, fill="x")
        self._create_styled_button(actions_frame, "View Billing History", self.show_billing_history).pack(
            pady=STYLE_CONFIG["WIDGET_PADY"] // 2, fill="x")
        self.stripe_portal_button = self._create_styled_button(actions_frame, "Manage Subscription (Stripe)",
                                                               self.open_stripe_portal, state="disabled")
        self.stripe_portal_button.pack(pady=STYLE_CONFIG["WIDGET_PADY"] // 2, fill="x")

    def show_frame(self, frame_name):
        if frame_name not in self.frames: print(f"Error: Frame '{frame_name}' not found."); return
        self.frames[frame_name].tkraise()
        if frame_name == "generate" and self.user_info and hasattr(self, 'generation_status_label') and self.frames[
            "generate"].winfo_exists():
            self.frames[frame_name].update_idletasks()
            input_area_width = self.frames[frame_name].winfo_width() * 0.4
            if hasattr(self.generation_status_label, 'configure'):
                self.generation_status_label.configure(wraplength=max(100, input_area_width - 40))

        if self.access_token:
            if hasattr(self, 'nav_frame'): self.nav_frame.grid()
        else:
            if hasattr(self, 'nav_frame'): self.nav_frame.grid_remove();
        if frame_name == "subscription":
            self.fetch_subscription_plans_and_user_status_async()
        elif frame_name == "gallery":
            self.load_gallery_async()
        elif frame_name == "profile":
            self.update_profile_display()

    def change_appearance_mode_event(self, new_appearance_mode):
        ctk.set_appearance_mode(new_appearance_mode)
        self.configure(fg_color=get_style("APP_BG_COLOR"))
        if hasattr(self, 'status_bar'): self.update_status(self.status_bar.cget("text"))

    def update_status(self, message, is_error=False):
        color_key = "ERROR_COLOR" if is_error else ("SUCCESS_COLOR" if "success" in message.lower() else "TEXT_COLOR")
        if hasattr(self, 'status_bar') and self.status_bar.winfo_exists():
            self.status_bar.configure(text=message, text_color=get_style(color_key))

    def _api_request_threaded(self, method, endpoint, data=None, json_data=None, files=None, params=None,
                              requires_auth=True, callback=None):
        def task():
            url = f"{self.fastapi_url}/{endpoint}"
            effective_headers = {}
            if hasattr(self, 'current_locale_for_api') and self.current_locale_for_api:
                effective_headers['Accept-Language'] = self.current_locale_for_api

            if requires_auth:
                if not self.access_token:
                    if callback: self.after(0, callback, {"error": "Authentication required.", "status_code": 401})
                    self.after(0, self.update_status, "Authentication required. Please login.", True)
                    self.after(0, lambda: (
                        self.show_frame("login"), self.nav_frame.grid_remove() if hasattr(self, 'nav_frame') else None))
                    return
                effective_headers['Authorization'] = f"Bearer {self.access_token}"

            response_data = None
            try:
                response = requests.request(method, url, headers=effective_headers, data=data, json=json_data,
                                            files=files, params=params, timeout=60)
                response.raise_for_status()
                if response.content:
                    try:
                        response_data = response.json()
                    except json.JSONDecodeError:
                        response_data = {"error": f"Non-JSON response: {response.text[:100]}",
                                         "status_code": response.status_code}
                else:
                    response_data = {}
            except requests.exceptions.HTTPError as e:
                error_detail = str(e)
                status_code_val = e.response.status_code if e.response else 500
                try:
                    error_detail = e.response.json().get("detail", str(e))
                except:
                    pass
                response_data = {"error": error_detail, "status_code": status_code_val}
                if status_code_val == 401: self.after(0, self.logout, True)
            except requests.exceptions.RequestException as e:
                response_data = {"error": f"Connection/Request Error: {str(e)}", "status_code": 503}
            except Exception as e:
                response_data = {"error": f"Unexpected error: {str(e)}", "status_code": 500}

            if callback:
                self.after(0, callback, response_data)

            if files:
                for f_key, f_tuple in files.items():
                    if hasattr(f_tuple, 'close'):
                        f_tuple.close()
                    elif isinstance(f_tuple, tuple) and len(f_tuple) > 1 and hasattr(f_tuple[1], 'close'):
                        f_tuple[1].close()

        threading.Thread(target=task, daemon=True).start()

    def save_token(self, token):
        self.access_token = token
        try:
            with open(".auth_token_v3", "w") as f:
                f.write(token)
        except Exception as e:
            print(f"Could not save token: {e}")

    def load_token(self):
        try:
            with open(".auth_token_v3", "r") as f:
                self.access_token = f.read().strip()
            if not self.access_token: self.access_token = None
        except FileNotFoundError:
            self.access_token = None
        except Exception as e:
            print(f"Could not load token: {e}");
            self.access_token = None

    def delete_token_file(self):
        try:
            if os.path.exists(".auth_token_v3"): os.remove(".auth_token_v3")
        except Exception as e:
            print(f"Could not delete token file: {e}")

    def login(self):
        username = self.login_username_entry.get()
        password = self.login_password_entry.get()
        if not username or not password:
            self.login_status_label.configure(text="Username and password required.",
                                              text_color=get_style("ERROR_COLOR"));
            return

        self.login_status_label.configure(text="Logging in...",
                                          text_color=get_style("TEXT_COLOR"))

        def _login_callback(response):
            if response and response.get("access_token"):
                self.save_token(response["access_token"])
                self.login_status_label.configure(text="")
                self.fetch_user_info_async(show_default_frame=True)
            else:
                detail = response.get("error", "Login failed.") if response else "Login failed."
                self.login_status_label.configure(text=detail, text_color=get_style("ERROR_COLOR"))
                self.update_status(detail, is_error=True)

        self._api_request_threaded("POST", "token", data={"username": username, "password": password},
                                   requires_auth=False, callback=_login_callback)

    def register(self):
        username = self.register_username_entry.get()
        email = self.register_email_entry.get()
        password = self.register_password_entry.get()
        if not username or not email or not password:
            self.register_status_label.configure(text="All fields required.",
                                                 text_color=get_style("ERROR_COLOR"));
            return

        self.register_status_label.configure(text="Registering...", text_color=get_style("TEXT_COLOR"))

        def _register_callback(response):
            if response and response.get("id") and not response.get("error"):
                self.register_status_label.configure(
                    text="Registration successful! Please verify your email and login.",
                    text_color=get_style("SUCCESS_COLOR"), wraplength=280)
                self.update_status("Registration successful! Please verify your email.")
            else:
                detail = response.get("error", "Registration failed.") if response else "Registration failed."
                self.register_status_label.configure(text=detail, text_color=get_style("ERROR_COLOR"))
                self.update_status(detail, is_error=True)

        self._api_request_threaded("POST", "register",
                                   json_data={"username": username, "email": email, "password": password},
                                   requires_auth=False, callback=_register_callback)

    def logout(self, show_login_frame=True):
        self.access_token = None;
        self.user_info = None;
        self.delete_token_file()
        self.update_status("Logged out successfully.")
        if hasattr(self, 'image_label'): self.image_label.configure(image=None,
                                                                    text="Your AI masterpiece(s) will appear here!")
        self.current_base_image_filename_server = None;
        self.current_base_image_for_img2img_local_path = None
        if hasattr(self, 'img2img_base_image_label'): self.img2img_base_image_label.configure(
            text="No base image selected.")
        for F_name in self.frames:
            if hasattr(self, f"{F_name}_status_label"): getattr(self, f"{F_name}_status_label").configure(text="")
        if hasattr(self, 'nav_frame'): self.nav_frame.grid_remove()
        if show_login_frame: self.show_frame("login")

    def fetch_user_info_async(self, show_default_frame=True):
        if not self.access_token:
            if show_default_frame: self.show_frame("login");
            if hasattr(self, 'nav_frame'): self.nav_frame.grid_remove()
            return

        def _fetch_user_callback(response):
            if response and not response.get("error"):
                self.user_info = response
                self.update_status(
                    f"Logged in: {self.user_info['username']} | Plan: {self.user_info['subscription_plan'].capitalize()} | Credits: {self.user_info['image_credits_remaining']}")
                self.update_profile_display()
                if hasattr(self, 'nav_frame'): self.nav_frame.grid()
                if show_default_frame: self.show_frame("generate")
                self._update_generate_button_state()
            else:
                error_msg = response.get("error", "Unknown error fetching user.") if response else "No response"
                self.update_status(f"Session error: {error_msg}. Please login again.", True)
                self.logout(show_login_frame=True)

        self._api_request_threaded("GET", "users/me", callback=_fetch_user_callback)

    def update_profile_display(self):
        if self.user_info:
            self.profile_username_label.configure(text=self.user_info.get('username', 'N/A'))
            self.profile_email_label.configure(text=self.user_info.get('email', 'N/A'))
            plan_name = self.user_info.get('subscription_plan', 'N/A').capitalize()
            plan_status = self.user_info.get('subscription_status', 'N/A').capitalize()
            self.profile_plan_label.configure(text=f"{plan_name} ({plan_status})")
            self.profile_credits_label.configure(text=str(self.user_info.get('image_credits_remaining', 'N/A')))
            expiry_iso = self.user_info.get('plan_expiry_date');
            expiry_display = "N/A"
            if expiry_iso:
                try:
                    expiry_display = datetime.fromisoformat(expiry_iso).strftime('%Y-%m-%d %H:%M UTC')
                except:
                    expiry_display = str(expiry_iso)
            elif plan_name.lower() == "free":
                expiry_display = "Perpetual (Free Tier)"
            self.profile_expiry_label.configure(text=expiry_display)
            self.profile_email_verified_label.configure(
                text="Verified" if self.user_info.get('email_verified') else "Not Verified",
                text_color=get_style("SUCCESS_COLOR") if self.user_info.get('email_verified') else get_style(
                    "ERROR_COLOR"))

            avatar_url = self.user_info.get('avatar_url')
            if avatar_url:
                avatar_filename_server = os.path.basename(avatar_url)
                local_avatar_path = os.path.join(AVATARS_DIR, avatar_filename_server)
                if os.path.exists(local_avatar_path):
                    try:
                        img = Image.open(local_avatar_path).resize((100, 100), Image.LANCZOS)
                        ctk_avatar = ctk.CTkImage(light_image=img, dark_image=img, size=(100, 100))
                        self.profile_avatar_image_label.configure(image=ctk_avatar, text="")
                    except Exception as e:
                        self.profile_avatar_image_label.configure(text="Avatar\nError", image=None);
                        print(
                            f"Avatar load error: {e}")
                else:
                    self.profile_avatar_image_label.configure(text="Avatar\n(Syncing)", image=None)
            else:
                self.profile_avatar_image_label.configure(text="No\nAvatar", image=None, font=(
                    STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_SMALL"]))

            can_manage_stripe = self.user_info.get(
                'stripe_customer_id') and STRIPE_CUSTOMER_PORTAL_LINK and "YOUR_STRIPE" not in STRIPE_CUSTOMER_PORTAL_LINK
            if hasattr(self, 'stripe_portal_button'): self.stripe_portal_button.configure(
                state="normal" if can_manage_stripe else "disabled")
        else:
            for attr in ["username", "email", "plan", "credits", "expiry", "email_verified"]:
                if hasattr(self, f"profile_{attr}_label"): getattr(self, f"profile_{attr}_label").configure(text="N/A")
            if hasattr(self, "profile_avatar_image_label"): self.profile_avatar_image_label.configure(text="No\nAvatar",
                                                                                                      image=None);
            if hasattr(self, "stripe_portal_button"): self.stripe_portal_button.configure(state="disabled")
        self._update_generate_button_state()

    def on_engine_selected(self, selected_engine):
        is_dalle = selected_engine == "dall-e-3"
        self.img2img_upload_button.configure(state="disabled" if is_dalle else "normal")
        self.img2img_strength_slider.configure(state="disabled" if is_dalle else "normal")
        if is_dalle:
            self.param_sliders["width"].set(1024);
            self.param_value_labels["width"].configure(text="1024px")
            self.param_sliders["height"].set(1024);
            self.param_value_labels["height"].configure(text="1024px")
        else:
            self.param_sliders["width"].set(512);
            self.param_value_labels["width"].configure(text="512px")
            self.param_sliders["height"].set(512);
            self.param_value_labels["height"].configure(text="512px")

    def upload_base_image_for_img2img(self):
        filepath = filedialog.askopenfilename(title="Select Base Image",
                                              filetypes=(
                                                  ("Image files", "*.png *.jpg *.jpeg *.webp"), ("All files", "*.*")))
        if not filepath: return

        self.current_base_image_for_img2img_local_path = filepath
        self.img2img_base_image_label.configure(text=os.path.basename(filepath))
        self.update_status(f"Base image '{os.path.basename(filepath)}' selected. Uploading to server...")

        def _upload_callback(response):
            if response and response.get("filename") and not response.get("error"):
                self.current_base_image_filename_server = response["filename"]
                self.update_status(f"Base image '{response['filename']}' ready on server.", False)
            else:
                self.current_base_image_for_img2img_local_path = None
                self.current_base_image_filename_server = None
                self.img2img_base_image_label.configure(text="Upload failed.")
                self.update_status(
                    f"Base image upload failed: {response.get('error', 'Unknown error') if response else 'No response'}",
                    True)

        with open(filepath, 'rb') as f_obj:
            files_data = {'file': (
                os.path.basename(filepath), f_obj, f'image/{os.path.splitext(filepath)[1].lstrip(".")}')}
            self._api_request_threaded("POST", "upload/img2img-base", files=files_data, callback=_upload_callback)

    def trigger_generate_image_thread(self):
        if not self.user_info: self.generation_status_label.configure(text="User not logged in.",
                                                                      text_color=get_style("ERROR_COLOR")); return
        if self.user_info.get('image_credits_remaining',
                              0) <= 0 and self.user_info.get('subscription_plan') != "free":
            self.generation_status_label.configure(text="No image credits remaining.",
                                                   text_color=get_style("ERROR_COLOR"));
            self.show_frame("subscription");
            return
        if self.user_info.get('subscription_status') != 'active' and self.user_info.get('subscription_plan') != "free":
            self.generation_status_label.configure(text="Subscription not active.",
                                                   text_color=get_style("ERROR_COLOR"));
            self.show_frame("subscription");
            return

        prompt_text = self.prompt_entry.get("1.0", "end-1c").strip()
        if not prompt_text: self.generation_status_label.configure(text="Prompt cannot be empty.",
                                                                   text_color=get_style("ERROR_COLOR")); return

        self.generation_status_label.configure(text="Preparing generation...", text_color=get_style("TEXT_COLOR"))
        self.progress_bar.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.progress_bar.start()
        self.generate_button_main.configure(state="disabled")

        payload = {
            "prompt": prompt_text,
            "negative_prompt": self.neg_prompt_entry.get("1.0", "end-1c").strip(),
            "engine_id": self.engine_id_menu.get(),
            "width": int(
                self.param_sliders["width"].get() // 64 * 64) if self.engine_id_menu.get() != "dall-e-3" else int(
                self.param_sliders["width"].get()),
            "height": int(
                self.param_sliders["height"].get() // 64 * 64) if self.engine_id_menu.get() != "dall-e-3" else int(
                self.param_sliders["height"].get()),
            "steps": int(self.param_sliders["steps"].get()),
            "cfg_scale": float(self.param_sliders["cfg"].get()),
            "seed": int(self.seed_entry.get() or "0"),
            "num_images": int(self.num_images_entry.get() or "1"),
            "base_image_filename": self.current_base_image_filename_server if self.engine_id_menu.get() != "dall-e-3" else None,
            "image_strength": float(
                self.img2img_strength_slider.get()) if self.engine_id_menu.get() != "dall-e-3" else None,
        }
        self._api_request_threaded("POST", "generate/image", json_data=payload,
                                   callback=self._update_generate_image_ui)

    def _update_generate_image_ui(self, response):
        self.progress_bar.stop();
        self.progress_bar.grid_forget()
        self.generate_button_main.configure(state="normal")

        if response and response.get("generated_images"):
            generated_files = response["generated_images"]
            if generated_files:
                self.generated_image_path = os.path.join(GENERATED_IMAGES_DIR, generated_files[-1]["filename"])
                self.generation_status_label.configure(text=f"{len(generated_files)} image(s) generated!",
                                                       text_color=get_style("SUCCESS_COLOR"))
                self.display_generated_image(self.generated_image_path)
                self.fetch_user_info_async(show_default_frame=False)
            else:
                self.generation_status_label.configure(text="Generation completed but no images returned.",
                                                       text_color="orange")

            if response.get("errors") and response["errors"]:
                errors_str = "; ".join(response["errors"])
                current_text = self.generation_status_label.cget("text")
                wraplength_val = 300
                if hasattr(self, 'frames') and "generate" in self.frames and hasattr(self.frames["generate"],
                                                                                     'input_scroll_area') and \
                        self.frames["generate"].input_scroll_area.winfo_exists():
                    wraplength_val = max(100, self.frames["generate"].input_scroll_area.winfo_width() - 40)

                self.generation_status_label.configure(text=f"{current_text} Issues: {errors_str[:200]}...",
                                                       text_color="orange",
                                                       wraplength=wraplength_val)
        elif response and response.get("error"):
            self.generation_status_label.configure(text=f"Generation Error: {response['error']}",
                                                   text_color=get_style("ERROR_COLOR"))
            if hasattr(self, 'image_label'): self.image_label.configure(image=None,
                                                                        text="Image generation failed.")
        else:
            self.generation_status_label.configure(text="Image generation failed or unexpected server response.",
                                                   text_color=get_style("ERROR_COLOR"))
            if hasattr(self, 'image_label'): self.image_label.configure(image=None, text="Image generation failed.")
        self._update_generate_button_state()

    def display_generated_image(self, image_path_local):
        try:
            if not os.path.exists(image_path_local):
                self.generation_status_label.configure(text=f"Error: Image file not found at {image_path_local}",
                                                       text_color=get_style("ERROR_COLOR"))
                if hasattr(self, 'image_label'): self.image_label.configure(text="Error: Image file not found.",
                                                                            image=None); return

            pil_image = Image.open(image_path_local)
            self.displayed_image = pil_image  # Store the PIL image

            if hasattr(self, 'image_label'):
                self.image_label.update_idletasks()
                label_width = self.image_label.winfo_width()
                label_height = self.image_label.winfo_height()

                if label_width < 50 or label_height < 50: label_width, label_height = 512, 512

                img_w, img_h = pil_image.size
                aspect_ratio = img_w / img_h if img_h > 0 else 1

                new_w, new_h = label_width, int(label_width / aspect_ratio) if aspect_ratio > 0 else label_width
                if new_h > label_height: new_h, new_w = label_height, int(label_height * aspect_ratio)
                if new_w <= 0: new_w = 1
                if new_h <= 0: new_h = 1

                resized_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
                ctk_image = ctk.CTkImage(light_image=resized_image, dark_image=resized_image, size=(new_w, new_h))

                self.image_label.configure(image=ctk_image, text="")
        except Exception as e:
            self.generation_status_label.configure(text=f"Error displaying image: {e}",
                                                   text_color=get_style("ERROR_COLOR"))
            if hasattr(self, 'image_label'): self.image_label.configure(text=f"Error displaying image.", image=None);
            print(f"Error displaying image: {e}")
        self._update_generate_button_state()

    def fetch_subscription_plans_and_user_status_async(self):
        if not self.user_info: self.fetch_user_info_async(show_default_frame=False)
        if not self.user_info: self.update_status("Login required.", True); return

        plan = self.user_info.get('subscription_plan', 'N/A').capitalize();
        status = self.user_info.get('subscription_status', 'N/A').capitalize();
        credits = self.user_info.get('image_credits_remaining', 'N/A');
        expiry = self.user_info.get('plan_expiry_date');
        expiry_str = datetime.fromisoformat(expiry).strftime('%Y-%m-%d') if expiry else (
            "Perpetual" if plan.lower() == 'free' else "N/A")

        wraplength_val = 500
        if hasattr(self, 'frames') and "subscription" in self.frames and self.frames["subscription"].winfo_exists():
            wraplength_val = self.frames["subscription"].winfo_width() - 40

        self.current_plan_info_label.configure(
            text=f"Current: {plan} ({status}) - Credits: {credits} - Valid: {expiry_str}",
            wraplength=wraplength_val)

        def _fetch_plans_callback(response):
            for widget in self.plans_and_packs_frame.winfo_children(): widget.destroy()
            if response and isinstance(response, dict) and not response.get("error"):
                row, col, max_cols = 0, 0, 3
                for item_id, details in response.items():
                    card_border_color = get_style("BORDER_COLOR")
                    if self.user_info and details['type'] == 'subscription' and \
                            self.user_info['subscription_plan'] == item_id and \
                            self.user_info['subscription_status'] == 'active':
                        card_border_color = get_style("ACCENT_COLOR")

                    card = self._create_styled_frame(self.plans_and_packs_frame, border_color=card_border_color)
                    card.grid(row=row, column=col, padx=STYLE_CONFIG["WIDGET_PADX"], pady=STYLE_CONFIG["WIDGET_PADY"],
                              sticky="nsew")
                    card.grid_rowconfigure(3, weight=1)

                    ctk.CTkLabel(card, text=details['name'], font=(
                        STYLE_CONFIG["FONT_FAMILY_HEADINGS"], STYLE_CONFIG["FONT_SIZE_LARGE"],
                        STYLE_CONFIG["FONT_WEIGHT_BOLD"]), text_color=get_style("ACCENT_COLOR") if details[
                                                                                                       'type'] == 'subscription' else get_style(
                        "TEXT_COLOR")).pack(pady=(15, 5), padx=10)
                    price_txt = f"${details['price']}" + ("/month" if details['type'] == 'subscription' else "")
                    ctk.CTkLabel(card, text=price_txt, font=(
                        STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_MEDIUM"] + 2,
                        STYLE_CONFIG["FONT_WEIGHT_BOLD"])).pack(pady=2)
                    cred_txt = f"{details['image_credits']} credits" + (
                        "/month" if details['type'] == 'subscription' else " (One-time)")
                    ctk.CTkLabel(card, text=cred_txt,
                                 font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"])).pack(pady=2)
                    if details.get('max_resolution') != "N/A": ctk.CTkLabel(card,
                                                                            text=f"Max Res: {details['max_resolution']}",
                                                                            font=(STYLE_CONFIG["FONT_FAMILY_MAIN"],
                                                                                  STYLE_CONFIG["FONT_SIZE_SMALL"]),
                                                                            text_color=get_style(
                                                                                "TEXT_DISABLED_COLOR")).pack(pady=2)

                    features_frame = ctk.CTkFrame(card, fg_color="transparent")
                    features_frame.pack(pady=(10, 5), padx=10, fill="x", expand=True)

                    card_width = card.winfo_width()
                    if card_width <= 10: card_width = 200

                    for feature in details['features']:
                        ctk.CTkLabel(features_frame, text=f"✓ {feature}",
                                     font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                                     justify="left", anchor="w",
                                     wraplength=card_width - 30).pack(anchor="w")

                    is_current = (self.user_info and details['type'] == 'subscription' and self.user_info[
                        'subscription_plan'] == item_id and self.user_info['subscription_status'] == 'active')
                    if is_current:
                        ctk.CTkLabel(card, text="CURRENT PLAN", font=(
                            STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"],
                            STYLE_CONFIG["FONT_WEIGHT_BOLD"]), text_color=get_style("SUCCESS_COLOR")).pack(
                            pady=(10, 15))
                    elif item_id != "free":
                        btn_text = "Subscribe" if details['type'] == 'subscription' else "Buy Pack"
                        self._create_styled_button(card, btn_text,
                                                   lambda i_id=item_id: self.initiate_payment(i_id, "stripe")).pack(
                            pady=(10, 15), padx=10)

                    col = (col + 1) % max_cols
                    if col == 0: row += 1
            elif response and response.get("error"):
                ctk.CTkLabel(self.plans_and_packs_frame, text=f"Error: {response['error']}").pack()
            else:
                ctk.CTkLabel(self.plans_and_packs_frame, text="Could not load plans.").pack()

        self._api_request_threaded("GET", "subscriptions/plans", requires_auth=False, callback=_fetch_plans_callback)

    def initiate_payment(self, item_id, payment_provider):
        self.update_status(
            f"Initiating payment for {SUBSCRIPTION_PLANS[item_id]['name']} via {payment_provider}...")
        payload = {"item_id": item_id, "payment_provider": payment_provider}

        def _payment_session_callback(response):
            if response and response.get("url") and payment_provider == 'stripe':
                webbrowser.open_new_tab(response['url'])
                self.update_status(f"Redirecting to Stripe for {item_id}. Complete payment in browser.", False)
                self.show_payment_pending_message(item_id, "Stripe", response.get("sessionId"))
            elif response and response.get("error"):
                self.update_status(f"Payment initiation failed: {response['error']}", True)
            else:
                self.update_status(f"Payment initiation failed or provider response not recognized.", True)

        self._api_request_threaded("POST", "subscriptions/create-payment-session", json_data=payload,
                                   callback=_payment_session_callback)

    def show_payment_pending_message(self, item_id, provider, session_id_or_tx_id):
        dialog = ctk.CTkToplevel(self);
        dialog.title("Payment Processing");
        dialog.geometry("400x250")
        dialog.transient(self);
        dialog.grab_set()
        dialog.configure(fg_color=get_style("APP_BG_COLOR"))

        ctk.CTkLabel(dialog,
                     text=f"Payment for {SUBSCRIPTION_PLANS[item_id]['name']} via {provider} is being processed.",
                     wraplength=380, font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                     text_color=get_style("TEXT_COLOR")).pack(pady=20)
        ctk.CTkLabel(dialog,
                     text="Please complete payment in the browser. Your account will update automatically upon confirmation (this may take a few moments).",
                     wraplength=380, font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                     text_color=get_style("TEXT_COLOR")).pack(pady=5)
        self._create_styled_button(dialog, "Check Status / Refresh Profile",
                                   lambda: [self.fetch_user_info_async(show_default_frame=False),
                                            self.fetch_subscription_plans_and_user_status_async(),
                                            dialog.destroy()]).pack(pady=15)
        self._create_styled_button(dialog, "Close", dialog.destroy, fg_color="transparent", border_width=1,
                                   border_color=get_style("BORDER_COLOR"), text_color=get_style("TEXT_COLOR")).pack(
            pady=5)

    def select_avatar_file(self):
        filepath = filedialog.askopenfilename(title="Select Avatar Image (.png, .jpg)",
                                              filetypes=(
                                                  ("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")))
        if not filepath: return
        self.update_status("Uploading avatar...")

        def _avatar_upload_callback(response):
            if response and not response.get("error") and response.get("avatar_url"):
                self.update_status("Avatar updated successfully!", False)
                self.fetch_user_info_async(show_default_frame=False)
            else:
                error_msg = response.get("error", "Unknown error") if response else "No response"
                self.update_status(f"Avatar upload failed: {error_msg}", True)

        with open(filepath, 'rb') as f_obj:
            files_data = {'file': (
                os.path.basename(filepath), f_obj, f'image/{os.path.splitext(filepath)[1].lstrip(".")}')}
            self._api_request_threaded("POST", "users/me/avatar", files=files_data, callback=_avatar_upload_callback)

    def show_billing_history(self):
        history_window = ctk.CTkToplevel(self);
        history_window.title("Billing History");
        history_window.geometry("750x550")
        history_window.transient(self);
        history_window.grab_set()
        history_window.configure(fg_color=get_style("APP_BG_COLOR"))

        self._create_section_label(history_window, "Your Payments and Subscriptions").pack(
            pady=STYLE_CONFIG["MAIN_PADY"])
        scroll_frame = ctk.CTkScrollableFrame(history_window, fg_color=get_style("FRAME_BG_COLOR"));
        scroll_frame.pack(expand=True, fill="both", padx=STYLE_CONFIG["WIDGET_PADX"], pady=STYLE_CONFIG["WIDGET_PADY"])

        def _billing_history_callback(response):
            if response and isinstance(response, list) and not (
                    response[0].get("error") if response else False):
                if not response: ctk.CTkLabel(scroll_frame, text="No billing history found.").pack(pady=10); return
                for payment in response:
                    date_str = datetime.fromisoformat(payment['created_at']).strftime(
                        '%Y-%m-%d %H:%M') if payment.get('created_at') else 'N/A'
                    text = (
                        f"Item: {payment.get('item_name', payment.get('item_id', 'N/A'))} | Amount: {payment['amount']:.2f} {payment['currency']} | "
                        f"Status: {payment['status']} | Gateway: {payment['payment_gateway']} | Date: {date_str} | TxID: {payment.get('transaction_id', 'N/A')[:20]}...")
                    ctk.CTkLabel(scroll_frame, text=text, wraplength=700, justify="left",
                                 font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_NORMAL"]),
                                 text_color=get_style("TEXT_COLOR")).pack(anchor="w", pady=3,
                                                                          padx=5)
            elif response and (isinstance(response, dict) and response.get("error") or (
                    isinstance(response, list) and response and response[0].get("error"))):
                error_msg = response.get("error") if isinstance(response, dict) else response[0].get("error")
                ctk.CTkLabel(scroll_frame, text=f"Error loading history: {error_msg}").pack(pady=10)
            else:
                ctk.CTkLabel(scroll_frame, text="Could not load billing history or unexpected response.").pack(pady=10)

        self._api_request_threaded("GET", "users/me/billing-history", callback=_billing_history_callback)

    def open_stripe_portal(self):
        self.update_status("Fetching Stripe Customer Portal link...")

        def _portal_link_callback(response):
            if response and response.get("portal_url") and not response.get("error"):
                webbrowser.open_new_tab(response["portal_url"])
                self.update_status("Stripe Customer Portal opened in browser.", False)
            else:
                error = response.get("error", "Failed to get portal link.") if response else "No response"
                self.update_status(f"Could not open Stripe Portal: {error}", True)

        self._api_request_threaded("GET", "subscriptions/manage-portal", callback=_portal_link_callback)

    def request_password_reset_action(self):
        email = self.reset_email_entry.get()
        if not email: self.reset_req_status_label.configure(text="Email required.",
                                                            text_color=get_style("ERROR_COLOR")); return
        self.reset_req_status_label.configure(text="Sending request...", text_color=get_style("TEXT_COLOR"))

        def _req_reset_callback(response):
            msg = response.get("message", response.get("error", "Request failed.")) if response else "Request failed."
            color_key = "SUCCESS_COLOR" if response and response.get("message") and not response.get(
                "error") else "ERROR_COLOR"
            self.reset_req_status_label.configure(text=msg, text_color=get_style(color_key))
            if color_key == "SUCCESS_COLOR": self.show_frame("password_reset_confirm")

        self._api_request_threaded("POST", "request-password-reset", json_data={"email": email},
                                   requires_auth=False, callback=_req_reset_callback)

    def confirm_password_reset_action(self):
        token = self.reset_token_entry.get()
        new_password = self.reset_new_password_entry.get()
        if not token or not new_password: self.reset_conf_status_label.configure(
            text="Token and new password required.", text_color=get_style("ERROR_COLOR")); return
        self.reset_conf_status_label.configure(text="Resetting password...", text_color=get_style("TEXT_COLOR"))

        def _conf_reset_callback(response):
            msg = response.get("message", response.get("error", "Reset failed.")) if response else "Reset failed."
            color_key = "SUCCESS_COLOR" if response and response.get("message") and not response.get(
                "error") else "ERROR_COLOR"
            self.reset_conf_status_label.configure(text=msg, text_color=get_style(color_key))
            if color_key == "SUCCESS_COLOR": self.show_frame("login")

        self._api_request_threaded("POST", "confirm-password-reset",
                                   json_data={"token": token, "new_password": new_password},
                                   requires_auth=False, callback=_conf_reset_callback)

    def load_gallery_async(self, page=1):
        if not self.access_token: return
        self.current_gallery_page = page
        for widget in self.gallery_scroll_frame.winfo_children(): widget.destroy()
        ctk.CTkLabel(self.gallery_scroll_frame, text="Loading gallery...",
                     font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_MEDIUM"]),
                     text_color=get_style("TEXT_DISABLED_COLOR")).grid(row=0, column=0, columnspan=5,
                                                                       pady=STYLE_CONFIG["MAIN_PADY"])

        params = {"page": page, "page_size": 15,
                  "sort_by": self.gallery_sort_by_var.get(),
                  "sort_order": self.gallery_sort_order_var.get()}
        if self.gallery_filter_prompt_entry.get(): params["filter_prompt"] = self.gallery_filter_prompt_entry.get()
        if self.gallery_filter_tags_entry.get(): params["filter_tags"] = self.gallery_filter_tags_entry.get()

        self._api_request_threaded("GET", "gallery/images", params=params, callback=self._process_gallery_response)

    def _process_gallery_response(self, response):
        for widget in self.gallery_scroll_frame.winfo_children(): widget.destroy()
        if response and response.get("items") is not None and not response.get("error"):
            gallery_data = response["items"];
            self.total_gallery_pages = response.get("num_pages", 1)
            self.gallery_page_label.configure(text=f"Page {self.current_gallery_page}/{self.total_gallery_pages}")
            self.gallery_prev_button.configure(state="normal" if self.current_gallery_page > 1 else "disabled")
            self.gallery_next_button.configure(
                state="normal" if self.current_gallery_page < self.total_gallery_pages else "disabled")
            if not gallery_data: ctk.CTkLabel(self.gallery_scroll_frame,
                                              text="Gallery is empty or no results for filter.").grid(row=0, column=0,
                                                                                                      columnspan=5,
                                                                                                      pady=20); return

            row, col, max_cols = 0, 0, 5
            thumb_base_w = self.gallery_scroll_frame.winfo_width() // max_cols - (
                    STYLE_CONFIG["WIDGET_PADX"] * 2)
            thumb_h = int(thumb_base_w * 0.75)
            thumb_size = (thumb_base_w if thumb_base_w > 50 else 120, thumb_h if thumb_h > 50 else 90)

            for img_data in gallery_data:
                card = self._create_styled_frame(self.gallery_scroll_frame, fg_color=get_style("INPUT_BG_COLOR"))
                card.grid(row=row, column=col, padx=STYLE_CONFIG["WIDGET_PADX"] // 2,
                          pady=STYLE_CONFIG["WIDGET_PADY"] // 2, sticky="nsew")
                card.bind("<Button-1>", lambda event, data=img_data, path=os.path.join(GENERATED_IMAGES_DIR, img_data[
                    'image_filename']): self.show_gallery_image_details(data, path))

                img_label = ctk.CTkLabel(card, text="", height=thumb_size[1],
                                         fg_color=get_style("FRAME_BG_COLOR"))
                img_label.pack(fill="x", padx=3, pady=3)

                info_frame = ctk.CTkFrame(card, fg_color="transparent")
                info_frame.pack(fill="x", padx=5, pady=(0, 5))
                prompt_snippet = (
                    img_data['prompt'][:30] + '...' if len(img_data['prompt']) > 30 else img_data['prompt'])
                ctk.CTkLabel(info_frame, text=prompt_snippet,
                             font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_SMALL"]),
                             wraplength=thumb_base_w - 10, anchor="w", text_color=get_style("TEXT_COLOR")).pack(
                    fill="x")

                engine_text = f"Engine: {img_data.get('engine_id', 'N/A').replace('stable-diffusion-', 'SD-')}"
                ctk.CTkLabel(info_frame, text=engine_text,
                             font=(STYLE_CONFIG["FONT_FAMILY_MAIN"], STYLE_CONFIG["FONT_SIZE_SMALL"] - 1),
                             text_color=get_style("TEXT_DISABLED_COLOR"), anchor="w").pack(fill="x")

                local_img_path = os.path.join(GENERATED_IMAGES_DIR, img_data['image_filename'])
                threading.Thread(target=self._load_gallery_thumbnail_threaded,
                                 args=(img_label, local_img_path, thumb_size), daemon=True).start()

                col = (col + 1) % max_cols
                if col == 0: row += 1
        elif response and response.get("error"):
            ctk.CTkLabel(self.gallery_scroll_frame, text=f"Error: {response['error']}").grid(row=0, column=0,
                                                                                             columnspan=5, pady=20)
        else:
            ctk.CTkLabel(self.gallery_scroll_frame, text="Could not load gallery.").grid(row=0, column=0, columnspan=5,
                                                                                         pady=20)

    def _load_gallery_thumbnail_threaded(self, label_widget, image_path, thumb_size):
        try:
            if not os.path.exists(image_path):
                self.after(0, label_widget.configure, {"text": "Missing", "image": None});
                return
            pil_img = Image.open(image_path);
            pil_img.thumbnail(thumb_size, Image.LANCZOS)
            ctk_thumb = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            self.after(0, label_widget.configure, {"image": ctk_thumb, "text": ""})
        except Exception as e:
            self.after(0, label_widget.configure, {"text": "Load Err", "image": None});
            print(f"Thumb err: {e}")

    def show_gallery_image_details(self, image_data, image_path_full):
        self.display_generated_image(image_path_full)
        self.prompt_entry.delete("1.0", "end");
        self.prompt_entry.insert("1.0", image_data.get('prompt', ''))
        self.neg_prompt_entry.delete("1.0", "end");
        self.neg_prompt_entry.insert("1.0", image_data.get('negative_prompt', ''))

        self.engine_id_menu.set(image_data.get('engine_id', self.engine_options[0]))
        self.on_engine_selected(self.engine_id_menu.get())

        self.param_sliders["width"].set(image_data.get('width', 512));
        self.param_value_labels["width"].configure(text=f"{image_data.get('width', 512)}px")
        self.param_sliders["height"].set(image_data.get('height', 512));
        self.param_value_labels["height"].configure(text=f"{image_data.get('height', 512)}px")
        self.param_sliders["steps"].set(image_data.get('steps', 30));
        self.param_value_labels["steps"].configure(text=str(image_data.get('steps', 30)))
        self.param_sliders["cfg"].set(image_data.get('cfg_scale', 7.0));
        self.param_value_labels["cfg"].configure(text=f"{image_data.get('cfg_scale', 7.0):.1f}")
        self.seed_entry.delete(0, "end");
        self.seed_entry.insert(0, str(image_data.get('seed', 0)))
        self.num_images_entry.delete(0, "end");
        self.num_images_entry.insert(0, "1")

        if image_data.get('base_image_filename') and self.engine_id_menu.get() != "dall-e-3":
            self.current_base_image_filename_server = image_data['base_image_filename']
            potential_local_base = os.path.join(IMG2IMG_BASE_DIR, image_data['base_image_filename'])
            if os.path.exists(potential_local_base):
                self.current_base_image_for_img2img_local_path = potential_local_base
                self.img2img_base_image_label.configure(text=os.path.basename(potential_local_base))
            else:
                self.img2img_base_image_label.configure(
                    text=f"Base: {image_data['base_image_filename']} (local N/A)")
        else:
            self.current_base_image_filename_server = None;
            self.current_base_image_for_img2img_local_path = None
            self.img2img_base_image_label.configure(text="No base image selected.")

        self.show_frame("generate")
        self.update_status(f"Loaded parameters from image: {os.path.basename(image_path_full)}")

    def apply_gallery_filters(self):
        self.load_gallery_async(page=1)

    def change_gallery_page(self, direction):
        new_page = self.current_gallery_page + direction
        if 1 <= new_page <= self.total_gallery_pages: self.load_gallery_async(page=new_page)

    def show_not_implemented_dialog(self, feature_name):
        messagebox.showinfo("Feature Not Implemented",
                            f"The '{feature_name}' feature is planned but not yet fully implemented.")

    def _update_generate_button_state(self):
        if not hasattr(self, 'user_info') or not self.user_info or not hasattr(self, 'generate_button_main'): return
        can_generate = (self.user_info.get('image_credits_remaining', 0) > 0 or self.user_info.get(
            'subscription_plan') == 'free') and self.user_info.get('subscription_status') == 'active'
        if hasattr(self, 'generate_button_main'):  # Check existence
            self.generate_button_main.configure(state="normal" if can_generate else "disabled")

        if hasattr(self, 'save_image_button'):
            if self.generated_image_path and os.path.exists(self.generated_image_path):
                self.save_image_button.configure(state="normal")
            else:
                self.save_image_button.configure(state="disabled")

    def on_closing(self):
        if messagebox.askokcancel("Quit GenMaster AI", "Are you sure you want to exit?"):
            self.destroy()


def run_fastapi_server(): uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    create_tables()
    threading.Thread(target=run_fastapi_server, daemon=True).start()
    print("Waiting for FastAPI server (3s)...");
    time.sleep(3);
    print("FastAPI started. Launching GUI.")
    app = AdvancedUltimateAIGeneratorApp(fastapi_url=BASE_URL)
    app.mainloop()