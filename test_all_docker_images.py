import asyncio
import aiohttp
import pandas as pd
from datasets import load_dataset
import csv
import os
import time
from typing import List, Tuple
from tqdm.asyncio import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


AUTH_KEY = "Bearer rlc-rtISTILowpUiVcLJjFHUmaYUJ3Y3C9ftGhO-wg"
BASE_URL = "https://toolbox.modal-origin.relace.run"

# Global semaphore to limit concurrent requests to 64
semaphore = asyncio.Semaphore(64)


async def test_toolbox_api(session, image_tag):
    """Test a single docker image with the toolbox API"""
    async with semaphore:  # Acquire semaphore before making requests
        try:
            # Create a new session
            async with session.post(
                f"{BASE_URL}/session/",
                json={"image_tag": image_tag},
                headers={
                    "Content-Type": "application/json",
                    "Authorization": AUTH_KEY,
                },
                timeout=aiohttp.ClientTimeout(total=120)
            ) as session_response:
                print(f"Testing {image_tag} - Session creation: {session_response.status}")
                
                if session_response.status == 200:
                    response_json = await session_response.json()
                    session_id = response_json["session_id"]

                    # Run tests for the session
                    async with session.post(
                        f"{BASE_URL}/session/{session_id}/test/",
                        headers={
                            "Authorization": AUTH_KEY,
                        },
                        timeout=aiohttp.ClientTimeout(total=120)
                    ) as test_response:
                        print(f"Testing {image_tag} - Test: {test_response.status}")
                        test_success = test_response.status == 200

                    # Delete the session
                    try:
                        async with session.delete(
                            f"{BASE_URL}/session/{session_id}",
                            headers={
                                "Authorization": AUTH_KEY,
                            },
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as delete_response:
                            print(f"Testing {image_tag} - Cleanup: {delete_response.status}")
                    except Exception as e:
                        print(f"Warning: Failed to cleanup session {session_id}: {e}")

                    return test_success
                else:
                    error_text = await session_response.text()
                    print(f"Failed to create session for {image_tag}: {error_text}")
                    return False
                    
        except asyncio.TimeoutError:
            print(f"Timeout testing {image_tag}")
            return False
        except Exception as e:
            print(f"Error testing {image_tag}: {e}")
            return False


async def process_image_with_progress(session, image_tag, progress_callback):
    """Process a single image and report progress"""
    success = await test_toolbox_api(session, image_tag)
    progress_callback(image_tag, success)
    return image_tag, success


def create_progress_callback(tested_images, output_file, total_images, pbar):
    """Create a callback function to track and save progress"""
    completed_count = [0]  # Use list to make it mutable in closure
    last_save_time = [time.time()]
    
    def callback(image_tag, success):
        tested_images[image_tag] = success
        completed_count[0] += 1
        
        status = "✅ WORKS" if success else "❌ FAILS"
        pbar.set_description(f"{status}: {image_tag[:50]}...")
        pbar.update(1)
        
        # Save progress every 10 seconds or every 50 images
        current_time = time.time()
        if current_time - last_save_time[0] > 10 or completed_count[0] % 50 == 0:
            with open(output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["docker_image", "success"])
                for img, succ in tested_images.items():
                    writer.writerow([img, succ])
            
            # Calculate statistics
            total_tested = len(tested_images)
            successful = sum(1 for s in tested_images.values() if s)
            failed = total_tested - successful
            success_rate = (successful / total_tested * 100) if total_tested > 0 else 0
            
            pbar.write(f"📊 Progress Update: {total_tested} tested | {successful} working | {failed} failed | {success_rate:.1f}% success rate")
            pbar.write(f"🔄 Active concurrent requests: {64 - semaphore._value}/{64}")
            
            last_save_time[0] = current_time
    
    return callback


def analyze_and_visualize_image_distribution(dataset, output_dir="."):
    """Analyze and visualize the distribution of docker images in the dataset."""
    print("📊 Analyzing docker image distribution...")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(dataset)
    
    # Get all unique docker images
    all_images = list(set(df["docker_image"]))
    
    print(f"🔍 Found {len(all_images)} unique docker images")
    print(f"📦 Total dataset entries: {len(df)}")
    
    # Extract base image names (everything before the ":")
    df['base_image'] = df['docker_image'].str.split(':').str[0]
    base_image_counts = df['base_image'].value_counts()
    
    print(f"\n📋 Docker Image Distribution by Base Names:")
    print(f"Total unique base images: {len(base_image_counts)}")
    for base_img, count in base_image_counts.items():
        print(f"  {base_img}: {count} examples")
    
    # Create visualizations
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pie chart
    colors = plt.cm.Set3(range(len(base_image_counts)))
    wedges, texts, autotexts = ax1.pie(base_image_counts.values, 
                                       labels=base_image_counts.index, 
                                       autopct='%1.1f%%',
                                       colors=colors,
                                       startangle=90)
    ax1.set_title('Docker Base Image Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    # Rotate labels if there are many
    if len(base_image_counts) > 8:
        for text in texts:
            text.set_fontsize(8)
            text.set_rotation(45)
    
    # Bar chart
    bars = ax2.bar(range(len(base_image_counts)), base_image_counts.values, color=colors)
    ax2.set_xlabel('Base Docker Images', fontweight='bold')
    ax2.set_ylabel('Number of Examples', fontweight='bold')
    ax2.set_title('Docker Base Image Distribution (Bar Chart)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(base_image_counts)))
    ax2.set_xticklabels(base_image_counts.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, base_image_counts.values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the visualization
    chart_file = os.path.join(output_dir, "initial_docker_image_distribution.png")
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n🖼️ Saved distribution charts to {chart_file}")
    
    # Show individual docker images (top 20)
    print(f"\n📝 Detailed Docker Image Distribution (top 20):")
    image_counts = df['docker_image'].value_counts().head(20)
    for img, count in image_counts.items():
        print(f"  {img}: {count} examples")
    
    return all_images


async def main():
    output_file = "docker_image_test_results.csv"
    tested_images = {}
    
    print("🚀 Starting Docker image testing with 64 concurrent requests...")

    # Load existing results if file exists
    if os.path.exists(output_file):
        try:
            df = pd.read_csv(output_file)
            tested_images.update(dict(zip(df["docker_image"], df["success"])))
            print(f"📂 Loaded {len(tested_images)} previously tested images")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")

    # Load the R2E-Gym dataset
    print("📦 Loading R2E-Gym dataset...")
    try:
        dataset = load_dataset("R2E-Gym/R2E-Gym-V1", split="train")
        print(f"📊 Dataset loaded with {len(dataset)} total entries")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    # Analyze and visualize image distribution
    all_images = analyze_and_visualize_image_distribution(dataset)

    # Filter out already tested images
    docker_images = [img for img in all_images if img not in tested_images]
    
    print(f"\n🆕 After filtering previously tested images:")
    print(f"   Images to test: {len(docker_images)}")
    print(f"   Previously tested: {len(tested_images)}")
    print(f"   Total unique images: {len(all_images)}")
    
    if not docker_images:
        print("✅ All images have already been tested!")
        return

    print(f"🚦 Using semaphore with 64 concurrent requests")
    
    start_time = time.time()
    
    # Create a single client session for all requests
    async with aiohttp.ClientSession() as session:
        # Create progress bar
        with tqdm(total=len(docker_images), desc="Testing Docker images", unit="image") as pbar:
            # Create progress callback
            progress_callback = create_progress_callback(tested_images, output_file, len(docker_images), pbar)
            
            # Create all tasks at once - semaphore will control concurrency
            tasks = [
                asyncio.create_task(
                    process_image_with_progress(session, image, progress_callback)
                )
                for image in docker_images
            ]
            
            pbar.write(f"📋 Created {len(tasks)} tasks, processing with max 64 concurrent requests...")
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    pbar.write(f"❌ Task error for {docker_images[i]}: {result}")
                    tested_images[docker_images[i]] = False

    # Save final results
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["docker_image", "success"])
        for image, success in tested_images.items():
            writer.writerow([image, success])

    # Final summary
    total_tested = len(tested_images)
    successful = sum(1 for success in tested_images.values() if success)
    failed = total_tested - successful
    success_rate = (successful / total_tested * 100) if total_tested > 0 else 0
    elapsed_time = time.time() - start_time
    
    print(f"\n🎉 Testing completed in {elapsed_time:.1f} seconds!")
    print(f"📊 Final Results:")
    print(f"   Total tested: {total_tested}")
    print(f"   ✅ Working: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success rate: {success_rate:.1f}%")
    print(f"   ⚡ Average time per image: {elapsed_time/len(docker_images):.2f}s")
    print(f"💾 Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())