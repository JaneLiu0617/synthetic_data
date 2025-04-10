import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import uuid
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import ollama # For LLM interaction
from ollama import Client # Import the Client class
import json
import time
import logging

# --- Configuration (Shared & Specific) ---

# Shared Config
ITEM_CATALOG = { # Keep this for statistical method & potential LLM grounding
    'Produce': {'Apple': 1.50, 'Banana': 0.50, 'Orange': 1.20, 'Broccoli': 2.50, 'Carrot': 1.80, 'Lettuce': 2.00},
    'Dairy': {'Milk': 3.50, 'Cheese': 5.00, 'Yogurt': 1.00, 'Butter': 4.00},
    'Bakery': {'Bread': 2.80, 'Croissant': 1.50, 'Bagel': 1.20},
    'Pantry': {'Pasta': 1.50, 'Rice': 2.00, 'Canned Soup': 1.80, 'Flour': 3.00, 'Sugar': 2.50},
    'Meat/Fish': {'Chicken Breast': 8.00, 'Ground Beef': 6.00, 'Salmon Fillet': 12.00},
    'Beverages': {'Soda Can': 1.00, 'Juice Box': 0.80, 'Water Bottle': 1.20},
    'Snacks': {'Chips': 3.00, 'Cookies': 2.50, 'Chocolate Bar': 1.50},
}
ALL_ITEMS = [(name, price, cat) for cat, items in ITEM_CATALOG.items() for name, price in items.items()]
STORE_NAMES = ['MegaMart', 'SuperGrocery', 'Local Foods', 'FreshMart']
STORE_LOCATIONS = ['Downtown', 'Uptown', 'Suburbia', 'Westside']
TAX_RATE = 0.08

# Simulation Parameters
N_CUSTOMERS_STATISTICAL = 50 # Number of customers for statistical method
N_WEEKS_STATISTICAL = 12     # Number of weeks for statistical method
N_RECEIPTS_LLM = 50          # Reduced number for LLM due to speed

# LLM Config
# OLLAMA_API_URL = "http://localhost:****/api/chat" # User provided URL - we need the host part
OLLAMA_HOST = "http://localhost:****" # Extract the host part for the Client
OLLAMA_MODEL = 'llama3.2-vision:latest' # <<< Changed model to llama3.2 >>>

# Logging setup for LLM attempts
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Ollama Client Initialization ---
# Initialize the Ollama client with the specific host URL
try:
    client = Client(host=OLLAMA_HOST)
    # Optional: Add a check here to see if the client can connect or list models
    # client.list()
    logging.info(f"Ollama client initialized for host: {OLLAMA_HOST}")
except Exception as e:
    logging.error(f"Failed to initialize Ollama client at {OLLAMA_HOST}: {e}")
    # Depending on the desired behavior, you might want to exit or raise the exception
    raise SystemExit(f"Could not connect to Ollama at {OLLAMA_HOST}")


# --- Part 1: Statistical Generator 

def generate_receipt_statistical(customer_id, week_num, start_date):
    """Generates a single synthetic grocery receipt using statistical methods."""
    receipt_id = f"STAT_{uuid.uuid4()}" # Add prefix
    store_name = random.choice(STORE_NAMES)
    store_location = random.choice(STORE_LOCATIONS)
    day_offset = random.randint(0, 6)
    receipt_date = start_date + timedelta(weeks=week_num, days=day_offset)
    receipt_time = timedelta(hours=random.randint(8, 20), minutes=random.randint(0, 59))
    receipt_datetime = receipt_date + receipt_time

    avg_items_per_receipt = 15
    std_dev_items = 5
    num_items_bought = max(1, int(np.random.normal(avg_items_per_receipt, std_dev_items)))

    items_purchased = []
    subtotal = 0.0

    preferred_categories = random.sample(list(ITEM_CATALOG.keys()), k=random.randint(1, 3))
    weights = [3 if item[2] in preferred_categories else 1 for item in ALL_ITEMS]
    weights = np.array(weights) / sum(weights)

    chosen_items_indices = np.random.choice(len(ALL_ITEMS), size=num_items_bought, p=weights, replace=True)
    chosen_items_sample = [ALL_ITEMS[i] for i in chosen_items_indices]

    for item_name, base_price, category in chosen_items_sample:
        quantity = random.randint(1, 3)
        unit_price = round(base_price * random.uniform(0.95, 1.05), 2)
        item_total = round(quantity * unit_price, 2)
        items_purchased.append({
            'receipt_id': receipt_id,
            'customer_id': customer_id,
            'item_name': item_name,
            'category': category,
            'quantity': quantity,
            'unit_price': unit_price,
            'item_total': item_total
        })
        subtotal += item_total

    subtotal = round(subtotal, 2)
    tax = round(subtotal * TAX_RATE, 2)
    total_amount = round(subtotal + tax, 2)

    receipt_summary = {
        'receipt_id': receipt_id,
        'customer_id': customer_id,
        'store_name': store_name,
        'store_location': store_location,
        'receipt_datetime': receipt_datetime,
        'week_num': week_num,
        'num_items': len(items_purchased),
        'subtotal': subtotal,
        'tax': tax,
        'total_amount': total_amount,
        'generator': 'Statistical' # Add generator type
    }
    return receipt_summary, items_purchased

def generate_statistical_data(num_customers, num_weeks, start_date=datetime.now()):
    """Generates the full dataset using the statistical method."""
    all_receipt_summaries = []
    all_items_purchased = []
    customer_ids = [f"CUST_S_{hashlib.sha1(str(i).encode()).hexdigest()[:8]}" for i in range(num_customers)]

    start_time = time.time()
    for customer_id in customer_ids:
        for week in range(num_weeks):
             if random.random() < 0.95:
                receipt_summary, items_purchased = generate_receipt_statistical(customer_id, week, start_date)
                all_receipt_summaries.append(receipt_summary)
                all_items_purchased.extend(items_purchased)

    receipts_df = pd.DataFrame(all_receipt_summaries)
    items_df = pd.DataFrame(all_items_purchased)
    if not receipts_df.empty:
        receipts_df['receipt_datetime'] = pd.to_datetime(receipts_df['receipt_datetime'])
    end_time = time.time()
    logging.info(f"Statistical generation took {end_time - start_time:.2f} seconds.")
    return receipts_df, items_df


# --- Part 2: LLM-Based Generator (Using Initialized Client) ---

def create_llm_prompt():
    """Creates the prompt for the LLM to generate one receipt."""
    categories = list(ITEM_CATALOG.keys())
    sample_items = [item[0] for item in random.sample(ALL_ITEMS, 5)] # Few examples

    prompt = f"""
    Generate a single, realistic JSON object representing a weekly grocery store receipt.

    Follow these instructions precisely:
    1.  Output *only* the JSON object, starting with {{ and ending with }}. Do not include any explanation or surrounding text.
    2.  The JSON object must have the following top-level keys:
        * "receipt_id": A unique string (e.g., "LLM_ followed by random characters).
        * "customer_id": A placeholder string like "CUST_L_XYZ".
        * "store_name": Choose one from {STORE_NAMES}.
        * "store_location": Choose one from {STORE_LOCATIONS}.
        * "receipt_datetime": A plausible ISO 8601 timestamp string (e.g., "2025-04-07T15:30:00"). Use a realistic date/time for a grocery shop in {datetime.now().year}.
        * "items": A JSON array of purchased items.
        * "subtotal": The sum of all item_total values (calculated by you, the LLM).
        * "tax": The subtotal multiplied by {TAX_RATE} (calculated by you, the LLM).
        * "total_amount": The sum of subtotal and tax (calculated by you, the LLM).
    3.  Each object in the "items" array must have the following keys:
        * "item_name": A common grocery item name (e.g., {', '.join(sample_items)}, Milk, Bread, Eggs, Chicken Breast, Apples, Pasta, Rice, etc.).
        * "category": Assign a plausible category from {categories}.
        * "quantity": An integer, usually between 1 and 5.
        * "unit_price": A plausible price (float) for the item (e.g., between 0.50 and 15.00).
        * "item_total": The result of quantity * unit_price (calculated by you, the LLM).
    4.  The receipt should contain between 5 and 25 items in the "items" array.
    5.  Ensure all calculations (item_total, subtotal, tax, total_amount) are mathematically correct based on the generated quantities and prices. Use two decimal places for currency values.

    Example JSON structure (values are illustrative):
    {{
      "receipt_id": "LLM_abc123",
      "customer_id": "CUST_L_1A3",
      "store_name": "MegaMart",
      "store_location": "Downtown",
      "receipt_datetime": "2025-04-07T17:45:12",
      "items": [
        {{
          "item_name": "Milk",
          "category": "Dairy",
          "quantity": 1,
          "unit_price": 3.55,
          "item_total": 3.55
        }},
        {{
          "item_name": "Bread",
          "category": "Bakery",
          "quantity": 1,
          "unit_price": 2.80,
          "item_total": 2.80
        }}
      ],
      "subtotal": 6.35,
      "tax": 0.51,
      "total_amount": 6.86
    }}

    Now, generate a new, valid JSON object for a grocery receipt. Only output the JSON.
    """
    return prompt

def parse_and_validate_llm_output(llm_response_content, attempt):
    """Attempts to parse LLM output, validates structure, recalculates totals."""
    try:
        # Basic cleanup - LLMs sometimes add markdown fences
        content = llm_response_content.strip().strip('```json').strip('```').strip()
        if not content: # Handle empty response
            logging.warning(f"LLM Attempt {attempt}: Received empty content.")
            return None, None
        receipt_json = json.loads(content)

        # Validate top-level keys
        required_keys = {"receipt_id", "customer_id", "store_name", "store_location",
                         "receipt_datetime", "items", "subtotal", "tax", "total_amount"}
        if not required_keys.issubset(receipt_json.keys()):
            logging.warning(f"LLM Attempt {attempt}: Missing top-level keys in JSON.")
            return None, None

        # Validate items structure and recalculate totals
        if not isinstance(receipt_json.get("items"), list) or not receipt_json.get("items"): # Use .get for safety
             logging.warning(f"LLM Attempt {attempt}: 'items' is not a non-empty list or missing.")
             return None, None

        item_keys = {"item_name", "category", "quantity", "unit_price", "item_total"}
        validated_items = []
        calculated_subtotal = 0.0

        for idx, item in enumerate(receipt_json["items"]):
            if not isinstance(item, dict) or not item_keys.issubset(item.keys()):
                logging.warning(f"LLM Attempt {attempt}: Item {idx} has invalid structure.")
                continue # Skip invalid items

            # Basic type validation/conversion
            try:
                quantity = int(item['quantity'])
                unit_price = round(float(item['unit_price']), 2)
                item_total_llm = round(float(item['item_total']), 2)
                item_name = str(item['item_name'])
                category = str(item.get('category', 'Unknown')) # Handle missing category gracefully
            except (ValueError, TypeError, KeyError) as e:
                 logging.warning(f"LLM Attempt {attempt}: Item {idx} has invalid data types or missing keys: {e}")
                 continue

            # *** Crucial: Recalculate item_total for consistency ***
            calculated_item_total = round(quantity * unit_price, 2)
            if abs(calculated_item_total - item_total_llm) > 0.01: # Allow tiny float difference
                logging.warning(f"LLM Attempt {attempt}: Item '{item_name}' - Recalculating item_total ({item_total_llm} -> {calculated_item_total})")

            validated_items.append({
                'receipt_id': receipt_json.get('receipt_id', f"LLM_MISSING_{uuid.uuid4()}"), # Handle missing ID
                'customer_id': receipt_json.get('customer_id', 'CUST_L_UNKNOWN'),
                'item_name': item_name,
                'category': category,
                'quantity': quantity,
                'unit_price': unit_price,
                'item_total': calculated_item_total # Use recalculated value
            })
            calculated_subtotal += calculated_item_total

        if not validated_items:
            logging.warning(f"LLM Attempt {attempt}: No valid items found after validation.")
            return None, None

        # *** Crucial: Recalculate subtotal, tax, total based on validated items ***
        calculated_subtotal = round(calculated_subtotal, 2)
        calculated_tax = round(calculated_subtotal * TAX_RATE, 2)
        calculated_total_amount = round(calculated_subtotal + calculated_tax, 2)

        receipt_datetime_obj = pd.to_datetime(receipt_json.get('receipt_datetime'), errors='coerce')

        # Create the final validated receipt summary using .get for safety
        receipt_summary = {
            'receipt_id': receipt_json.get('receipt_id', f"LLM_MISSING_{uuid.uuid4()}"),
            'customer_id': receipt_json.get('customer_id', 'CUST_L_UNKNOWN'),
            'store_name': receipt_json.get('store_name', 'Unknown Store'),
            'store_location': receipt_json.get('store_location', 'Unknown Location'),
            'receipt_datetime': receipt_datetime_obj,
            'week_num': receipt_datetime_obj.isocalendar().week if pd.notna(receipt_datetime_obj) else None,
            'num_items': len(validated_items),
            'subtotal': calculated_subtotal, # Use recalculated
            'tax': calculated_tax,           # Use recalculated
            'total_amount': calculated_total_amount, # Use recalculated
            'generator': 'LLM' # Add generator type
        }

        # Final check for valid datetime
        if pd.isna(receipt_summary['receipt_datetime']):
             logging.warning(f"LLM Attempt {attempt}: Invalid or missing datetime format.")
             # Decide if this is critical - maybe allow receipt with null date? For now, fail it.
             return None, None

        return receipt_summary, validated_items

    except json.JSONDecodeError as e:
        logging.error(f"LLM Attempt {attempt}: Failed to decode JSON: {e}")
        logging.debug(f"LLM Raw Output: {llm_response_content}")
        return None, None
    except Exception as e:
        logging.error(f"LLM Attempt {attempt}: Unexpected error during parsing: {e}")
        logging.debug(f"LLM Raw Output: {llm_response_content}")
        return None, None


def generate_llm_data(num_receipts):
    """Generates dataset by repeatedly calling the LLM using the initialized client."""
    all_receipt_summaries = []
    all_items_purchased = []
    successful_generations = 0
    attempts = 0
    max_attempts_per_receipt = 3 # Try a few times if LLM fails

    start_time = time.time()
    while successful_generations < num_receipts and attempts < num_receipts * max_attempts_per_receipt:
        attempt_num = attempts + 1
        logging.info(f"Attempting LLM generation {attempt_num} (Target: {successful_generations}/{num_receipts})...")
        prompt = create_llm_prompt()
        try:
            # <<< Use the initialized client instance >>>
            response = client.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.7} # Adjust temperature for creativity vs coherence
            )
            content = response['message']['content']

            receipt_summary, items_purchased = parse_and_validate_llm_output(content, attempt_num)

            if receipt_summary and items_purchased:
                all_receipt_summaries.append(receipt_summary)
                all_items_purchased.extend(items_purchased)
                successful_generations += 1
                logging.info(f"LLM Attempt {attempt_num}: Successfully generated and validated receipt {successful_generations}/{num_receipts}.")
            else:
                 logging.warning(f"LLM Attempt {attempt_num}: Failed validation or parsing.")

        except Exception as e:
            # Catch exceptions during the API call specifically
            logging.error(f"LLM Attempt {attempt_num}: Error during Ollama API call to {OLLAMA_HOST}: {e}")
            # Optional: Add a small delay before retrying
            time.sleep(2) # Increased sleep time slightly
        finally:
             attempts += 1


    end_time = time.time()
    logging.info(f"LLM generation loop finished. Got {successful_generations}/{num_receipts} receipts in {end_time - start_time:.2f} seconds ({attempts} total attempts).")

    receipts_df = pd.DataFrame(all_receipt_summaries)
    items_df = pd.DataFrame(all_items_purchased)
    # Datetime conversion already handled during parsing

    return receipts_df, items_df

# --- Part 3: Comparative Evaluation (No changes needed in functions) ---

def evaluate_fidelity_comparison(df_stat, df_llm, items_stat, items_llm):
    """Compares statistical similarity between Statistical and LLM datasets."""
    print("\n--- Fidelity Evaluation (Statistical vs LLM) ---")
    if df_stat.empty or df_llm.empty:
        print(" One or both datasets are empty. Cannot perform fidelity comparison.")
        return

    features_to_compare = ['num_items', 'subtotal', 'total_amount']

    # 1. Marginal Distributions
    print("\n1. Comparing Distributions:")
    fig, axes = plt.subplots(1, len(features_to_compare), figsize=(18, 5))
    fig.suptitle("Receipt Level Distributions (Statistical vs LLM)")
    for i, feature in enumerate(features_to_compare):
        # Check if feature exists and has data in both dataframes
        if feature in df_stat and not df_stat[feature].isnull().all() and \
           feature in df_llm and not df_llm[feature].isnull().all():
            sns.histplot(df_stat[feature], ax=axes[i], color='blue', label='Statistical', kde=True, stat="density", alpha=0.6)
            sns.histplot(df_llm[feature], ax=axes[i], color='green', label='LLM', kde=True, stat="density", alpha=0.6)
            axes[i].set_title(f"Distribution of {feature}")
            axes[i].legend()
            # KS Test
            try:
                # Drop NaNs before KS test for robustness
                stat_data = df_stat[feature].dropna()
                llm_data = df_llm[feature].dropna()
                if len(stat_data) > 1 and len(llm_data) > 1: # Need at least 2 points for KS test
                    ks_stat, p_val = stats.ks_2samp(stat_data, llm_data)
                    print(f"  KS test for '{feature}': Stat={ks_stat:.4f}, p-value={p_val:.4f} {'(Distributions differ significantly)' if p_val < 0.05 else '(Distributions similar)'}")
                else:
                    print(f"  Not enough data points for KS test for '{feature}'.")
            except Exception as e:
                print(f"  Could not run KS test for '{feature}': {e}")
        else:
            axes[i].set_title(f"{feature}\n(Data Missing/Invalid)")
            print(f"  Skipping distribution plot/test for '{feature}' due to missing data.")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


    # 2. Item Level Frequencies
    print("\n2. Comparing Item Frequencies:")
    if items_stat.empty or items_llm.empty:
         print(" Item data missing for one or both methods.")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Item Level Frequencies (Statistical vs LLM)")
        top_n = 15
        stat_item_counts = items_stat['item_name'].value_counts().nlargest(top_n)
        llm_item_counts = items_llm['item_name'].value_counts().nlargest(top_n)
        if not stat_item_counts.empty:
             stat_item_counts.plot(kind='bar', ax=axes[0], title=f'Top {top_n} Items (Statistical)', color='blue', alpha=0.7)
             axes[0].tick_params(axis='x', rotation=45, labelsize=8)
             print(f"  Top {top_n} items (Statistical): {list(stat_item_counts.index)}")
        else:
             axes[0].set_title(f'Top {top_n} Items (Statistical)\nNo Data')
             print("  No item data for Statistical method.")

        if not llm_item_counts.empty:
            llm_item_counts.plot(kind='bar', ax=axes[1], title=f'Top {top_n} Items (LLM)', color='green', alpha=0.7)
            axes[1].tick_params(axis='x', rotation=45, labelsize=8)
            print(f"  Top {top_n} items (LLM): {list(llm_item_counts.index)}")
        else:
            axes[1].set_title(f'Top {top_n} Items (LLM)\nNo Data')
            print("  No item data for LLM method.")


        if not stat_item_counts.empty and not llm_item_counts.empty:
            common_items = len(set(stat_item_counts.index) & set(llm_item_counts.index))
            print(f"  Common items in Top {top_n}: {common_items}/{top_n}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


    # 3. Category Distribution
    print("\n3. Comparing Category Frequencies:")
    if items_stat.empty or items_llm.empty:
         print(" Item data missing for one or both methods.")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Category Frequencies (Statistical vs LLM)")
        stat_cat_counts = items_stat['category'].value_counts()
        llm_cat_counts = items_llm['category'].value_counts()

        if not stat_cat_counts.empty:
            stat_cat_counts.plot(kind='bar', ax=axes[0], title='Categories (Statistical)', color='blue', alpha=0.7)
            axes[0].tick_params(axis='x', rotation=45, labelsize=9)
        else:
             axes[0].set_title('Categories (Statistical)\nNo Data')

        if not llm_cat_counts.empty:
            llm_cat_counts.plot(kind='bar', ax=axes[1], title='Categories (LLM)', color='green', alpha=0.7)
            axes[1].tick_params(axis='x', rotation=45, labelsize=9)
        else:
             axes[1].set_title('Categories (LLM)\nNo Data')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        print("  Category distribution comparison visually.")

def run_utility_analysis(receipts_df, items_df, label):
    """Runs standard utility analyses on a given dataset."""
    print(f"\n--- Utility Evaluation ({label}) ---")
    if receipts_df.empty: # Removed items_df check here, as some analyses only need receipts
        print(" Receipt data is empty. Cannot perform utility analysis.")
        return False # Indicate failure

    print(f"\nAnalysing {label} Data ({len(receipts_df)} receipts)...")
    analysis_possible = False # Track if at least one analysis runs

    # 1. Average Spending (Overall and Per Customer if possible)
    print(f"\n1. Spending Analysis ({label}):")
    if 'total_amount' in receipts_df.columns and not receipts_df['total_amount'].isnull().all():
        avg_receipt_total = receipts_df['total_amount'].mean()
        median_receipt_total = receipts_df['total_amount'].median()
        print(f"  Overall Avg Receipt Total: ${avg_receipt_total:.2f}")
        print(f"  Overall Median Receipt Total: ${median_receipt_total:.2f}")
        analysis_possible = True
    else:
        print("  Skipping spending analysis (total_amount missing or all null).")


    # Weekly spending requires consistent customer IDs and week numbers
    if 'customer_id' in receipts_df.columns and \
       'week_num' in receipts_df.columns and \
       'total_amount' in receipts_df.columns:
        try:
            # Filter out potential NaNs
            receipts_df_filtered = receipts_df.dropna(subset=['customer_id', 'week_num', 'total_amount'])
            # Need multiple customers and weeks to make this meaningful
            if receipts_df_filtered['customer_id'].nunique() > 1 and receipts_df_filtered['week_num'].nunique() > 1:
                 weekly_spending = receipts_df_filtered.groupby(['customer_id', 'week_num'])['total_amount'].sum().reset_index()
                 if not weekly_spending.empty:
                     avg_weekly_spending_per_cust = weekly_spending.groupby('customer_id')['total_amount'].mean()
                     plt.figure(figsize=(10, 4))
                     sns.histplot(avg_weekly_spending_per_cust, kde=True)
                     plt.title(f'Distribution of Average Weekly Spending per Customer ({label})')
                     plt.xlabel('Average Weekly Spend ($)')
                     plt.ylabel('Number of Customers')
                     plt.show()
                     print(f"  Avg Weekly Spend per Customer (mean): ${avg_weekly_spending_per_cust.mean():.2f}")
                     print(f"  Avg Weekly Spend per Customer (median): ${avg_weekly_spending_per_cust.median():.2f}")
                     analysis_possible = True
                 else: print("  Not enough valid weekly data per customer to plot weekly spending distribution.")
            else:
                 print("  Skipping weekly per-customer analysis (not enough distinct customers/weeks).")
        except Exception as e:
            print(f"  Could not calculate weekly spending per customer: {e}")
    else:
        print("  Skipping weekly per-customer analysis (requires customer_id, week_num, total_amount).")


    # 2. Popular Items/Categories
    print(f"\n2. Popular Items & Categories ({label}):")
    if items_df.empty:
         print("  Skipping item/category analysis (item data missing).")
    else:
        if 'item_name' in items_df.columns:
             top_items = items_df['item_name'].value_counts().nlargest(10)
             print(f"  Top 10 Purchased Items ({label}):")
             print(top_items)
             analysis_possible = True
        else: print("  Item name column missing.")

        if 'category' in items_df.columns:
            top_categories = items_df['category'].value_counts().nlargest(5)
            print(f"\n  Top 5 Purchased Categories ({label}):")
            print(top_categories)
            analysis_possible = True
        else: print("  Category column missing.")

    # 3. Shopping Day Trend
    print(f"\n3. Day of Week Trend ({label}):")
    if 'receipt_datetime' in receipts_df.columns and not receipts_df['receipt_datetime'].isnull().all():
        # Ensure datetime type after potential nulls
        receipts_df['receipt_datetime'] = pd.to_datetime(receipts_df['receipt_datetime'], errors='coerce')
        valid_dates = receipts_df.dropna(subset=['receipt_datetime'])
        if not valid_dates.empty:
             valid_dates['day_of_week'] = valid_dates['receipt_datetime'].dt.day_name()
             shopping_days = valid_dates['day_of_week'].value_counts().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
             ], fill_value=0)
             plt.figure(figsize=(10, 4))
             shopping_days.plot(kind='bar')
             plt.title(f'Number of Receipts by Day of Week ({label})')
             plt.xlabel('Day of Week')
             plt.ylabel('Number of Receipts')
             plt.xticks(rotation=45)
             plt.show()
             print(f"\n  Shopping Trend by Day of Week ({label}):")
             print(shopping_days)
             analysis_possible = True
        else:
             print("  Skipping day of week analysis (no valid dates).")
    else:
        print("  Skipping day of week analysis (receipt_datetime missing or invalid).")

    return analysis_possible # Return true if at least one analysis ran

def evaluate_privacy_comparison(receipts_df, items_df, label):
    """Performs basic privacy checks, adapted for comparison."""
    print(f"\n--- Privacy Evaluation ({label}) ---")
    if receipts_df.empty:
        print(" Receipt data is empty. Cannot perform privacy analysis.")
        return

    # 1. Check for Unique Identifiers (beyond intended keys)
    potential_leaks = []
    if len(receipts_df) > 1: # Only check if more than one row
        for col in receipts_df.columns:
            # Check if >95% are unique, excluding known unique keys and generator label
            if col not in ['receipt_id', 'generator']:
                try:
                    nunique = receipts_df[col].nunique()
                    # Use a threshold, e.g. if uniqueness > 0.95 and more than 10 unique values
                    if nunique / len(receipts_df) > 0.95 and nunique > 10 :
                        potential_leaks.append(col)
                except TypeError: # Handle non-hashable types if any sneak in
                     pass # Ignore columns that can't be checked for uniqueness easily

    if potential_leaks:
        print(f"  ({label}) Warning: Columns potentially revealing unique records (high uniqueness): {potential_leaks}")
    else:
        print(f"  ({label}) Basic Check: No obvious unintended high-uniqueness columns found.")

    # 2. Check for Outliers / Extremes
    if 'total_amount' in receipts_df.columns and not receipts_df['total_amount'].isnull().all():
        print(f"  ({label}) Max total amount: ${receipts_df['total_amount'].max():.2f}")
    if 'num_items' in receipts_df.columns and not receipts_df['num_items'].isnull().all():
        print(f"  ({label}) Max number of items: {int(receipts_df['num_items'].max())}")

    # 3. PII Presence
    print(f"  ({label}) PII Check: Both generators designed NOT to include direct PII.")
    print(f"  ({label}) Customer IDs: {'Hashed (Statistical)' if label == 'Statistical' else 'Placeholder/Generated (LLM)'}")

    # 4. Conceptual Risks (General Comments)
    if label == 'LLM':
        print(f"  ({label}) Conceptual: LLM generation might be less predictable. Output depends heavily on the prompt and model's training data (base Llama models are generally trained on public web data). Risk of memorization/specificity is theoretically higher than pure statistical sampling IF the LLM was trained inappropriately, but lower if it generalizes well.")
    else:
        print(f"  ({label}) Conceptual: Statistical generation is more controlled based on defined rules and sampling. Less likely to produce highly specific/unique outputs not covered by the rules.")


# --- Main Comparison Execution ---

if __name__ == "__main__":
    print("--- Synthetic Data Generation Comparison ---")
    print(f"Using Ollama Model: {OLLAMA_MODEL} at {OLLAMA_HOST}")

    # --- Generate Data ---
    print("\n1. Generating data using STATISTICAL method...")
    stat_receipts_df, stat_items_df = generate_statistical_data(
        N_CUSTOMERS_STATISTICAL, N_WEEKS_STATISTICAL, start_date=datetime(2025, 1, 1) # Use consistent future start date
        )
    print(f"Statistical method generated: {len(stat_receipts_df)} receipts, {len(stat_items_df)} item lines.")

    print(f"\n2. Generating data using LLM ({OLLAMA_MODEL})... (Target: {N_RECEIPTS_LLM} receipts)")
    print("NOTE: LLM generation can be slow. Please be patient.")
    llm_receipts_df, llm_items_df = generate_llm_data(N_RECEIPTS_LLM)
    print(f"LLM method generated: {len(llm_receipts_df)} receipts, {len(llm_items_df)} item lines.")

    # --- Evaluate & Compare ---
    print("\n--- Starting Comparative Evaluation ---")

    # 3. Fidelity Comparison
    evaluate_fidelity_comparison(stat_receipts_df, llm_receipts_df, stat_items_df, llm_items_df)

    # 4. Utility Comparison (Run analysis on both datasets)
    stat_utility_ok = run_utility_analysis(stat_receipts_df, stat_items_df, "Statistical")
    llm_utility_ok = run_utility_analysis(llm_receipts_df, llm_items_df, "LLM")

    # 5. Privacy Comparison (Run checks on both datasets)
    evaluate_privacy_comparison(stat_receipts_df, stat_items_df, "Statistical")
    evaluate_privacy_comparison(llm_receipts_df, llm_items_df, "LLM")

    # --- Conclusion ---
    print("\n--- Comparison Summary ---")

    print("\nGeneration Speed & Reliability:")
    # Speed was logged during generation. Reliability based on success rate.
    stat_target = N_CUSTOMERS_STATISTICAL * N_WEEKS_STATISTICAL # Approx target, not accounting for skip chance
    stat_success_rate = len(stat_receipts_df) / stat_target if stat_target > 0 else 1
    llm_success_rate = len(llm_receipts_df) / N_RECEIPTS_LLM if N_RECEIPTS_LLM > 0 else 0
    print(f"- Statistical: Relatively Fast. Reliability based on code logic (Actual/Target ~{stat_success_rate:.1%}).") # Simplistic rate
    print(f"- LLM ({OLLAMA_MODEL}): Significantly Slower. Reliability depends on LLM adherence to prompt & parsing robustness (Actual Receipts/Target Receipts = {llm_success_rate:.1%}).")

    print("\nFidelity (Statistical Similarity):")
    print("- Review the distribution plots and KS test results generated above.")
    print("- Review the item/category frequency lists and plots.")
    print(f"- LLM ({OLLAMA_MODEL}) output quality depends heavily on the model's capabilities and the prompt effectiveness.")

    print("\nUtility (Usefulness for Analysis):")
    print(f"- Statistical data utility analysis ran successfully (at least partially): {stat_utility_ok}")
    print(f"- LLM data utility analysis ran successfully (at least partially): {llm_utility_ok}")
    print("- Check if analyses yielded plausible results for both. LLM data might have more variance or unexpected patterns.")
    print("- Statistical method easily preserves customer weekly patterns. LLM method (as implemented) generates independent receipts, limiting longitudinal analysis.")

    print("\nPrivacy (Basic Checks):")
    print("- Both methods avoided direct PII by design.")
    print("- Review outlier values and potential uniqueness risks noted above.")
    print("- Statistical method offers more control. LLM's output predictability is lower.")

    print("\nOverall Recommendation:")
    print("- **Statistical Method:** Still generally preferred for large-scale, consistent, structured data with known constraints and relationships (like weekly habits). Faster, reliable, controllable.")
    print(f"- **LLM Method ({OLLAMA_MODEL}):** Can generate diverse examples but faces challenges with speed, numerical/structural consistency, and maintaining relationships across records without advanced techniques. Parsing/validation is essential. Best suited for augmenting data or generating less structured/textual elements, or when creative variance is desired over strict consistency.")

    print("\n--- Comparison Complete ---")