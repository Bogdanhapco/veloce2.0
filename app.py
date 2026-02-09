"""
LTX-2 19B Distilled Model - API Test Script
This tests the specific parameters that LTX-2 typically expects
"""

from gradio_client import Client

GRADIO_URL = "https://5a84f7d44bed240468.gradio.live"

print("🔍 Connecting to LTX-2 19B Distilled model...")
client = Client(GRADIO_URL)
print("✅ Connected!\n")

print("=" * 70)
print("API STRUCTURE:")
print("=" * 70)
api_info = client.view_api()
print(api_info)
print("\n")

print("=" * 70)
print("TESTING LTX-2 PARAMETER COMBINATIONS:")
print("=" * 70)

# Test 1: Minimal LTX-2 call
print("\n🧪 TEST 1: Just prompt (simplest)")
print("Code: client.predict(prompt, fn_index=0)")
try:
    result = client.predict(
        "a cat playing piano",
        fn_index=0
    )
    print(f"✅ SUCCESS!")
    print(f"Result: {result}")
    print("\n⭐ WINNER! Use this:")
    print('result = client.predict(prompt, fn_index=0)')
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 2: LTX-2 with negative prompt
print("\n🧪 TEST 2: Prompt + negative prompt")
print("Code: client.predict(prompt, negative_prompt, fn_index=0)")
try:
    result = client.predict(
        "a cat playing piano",  # prompt
        "",                      # negative_prompt (empty)
        fn_index=0
    )
    print(f"✅ SUCCESS!")
    print(f"Result: {result}")
    print("\n⭐ WINNER! Use this:")
    print('result = client.predict(prompt, "", fn_index=0)')
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 3: LTX-2 typical parameters
print("\n🧪 TEST 3: Full LTX-2 parameters")
print("Code: prompt, neg_prompt, num_frames, width, height, guidance, steps, seed")
try:
    result = client.predict(
        "a cat playing piano",  # prompt
        "",                      # negative_prompt
        121,                     # num_frames (LTX-2 default is often 121)
        768,                     # width
        512,                     # height
        3.0,                     # guidance_scale
        30,                      # num_inference_steps
        42,                      # seed
        fn_index=0
    )
    print(f"✅ SUCCESS!")
    print(f"Result: {result}")
    print("\n⭐ WINNER! Use full parameters")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 4: Without seed
print("\n🧪 TEST 4: Without seed parameter")
try:
    result = client.predict(
        "a cat playing piano",
        "",      # negative_prompt
        121,     # num_frames
        768,     # width
        512,     # height
        3.0,     # guidance_scale
        30,      # num_inference_steps
        fn_index=0
    )
    print(f"✅ SUCCESS!")
    print(f"Result: {result}")
    print("\n⭐ WINNER! No seed needed")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 5: Different frame counts
print("\n🧪 TEST 5: Try with 97 frames (5 seconds)")
try:
    result = client.predict(
        "a cat playing piano",
        "",      # negative_prompt
        97,      # num_frames (5 sec at 24fps)
        768,     # width
        512,     # height
        3.0,     # guidance_scale
        30,      # num_inference_steps
        fn_index=0
    )
    print(f"✅ SUCCESS!")
    print(f"Result: {result}")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 6: Check if there are multiple endpoints
print("\n🧪 TEST 6: Try fn_index=1")
try:
    result = client.predict("a cat playing piano", fn_index=1)
    print(f"✅ SUCCESS!")
    print(f"Result: {result}")
    print("\n⭐ Use fn_index=1 instead!")
except Exception as e:
    print(f"❌ Failed: {e}")

print("\n" + "=" * 70)
print("📊 SUMMARY:")
print("=" * 70)
print("""
Look for tests marked with "✅ SUCCESS!" above.
That's the exact code you need to use in your Streamlit app!

Common LTX-2 19B parameter structures:
1. Just prompt
2. Prompt + negative_prompt  
3. Prompt + negative_prompt + num_frames + width + height + guidance + steps
4. Same as #3 but with seed at the end

Copy the working code from the successful test above! ⬆️
""")

print("\n💡 QUICK TIP:")
print("If ALL tests failed, try opening the Gradio URL in your browser")
print("and look at what input fields are available. Match those exactly!")
print(f"\nYour URL: {GRADIO_URL}")
