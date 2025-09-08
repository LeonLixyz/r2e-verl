import asyncio
import aiohttp
import logging
import sys
import os

# Add the verl directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from verl.utils.http_client import http_request, SharedHTTPClient

logging.basicConfig(level=logging.INFO)  # Changed to INFO to reduce noise
logger = logging.getLogger(__name__)

async def create_session(base_url: str, headers: dict) -> str:
    """Create a new toolbox session using shared HTTP client."""
    try:
        async with http_request('POST', f"{base_url}/session/", headers=headers, timeout=30) as response:
            if response.status == 200:
                data = await response.json()
                session_id = data["session_id"]
                logger.debug(f"Created toolbox session: {session_id}")
                return session_id
            else:
                error_text = await response.text()
                raise Exception(f"Failed to create session (status {response.status}): {error_text}")
    except Exception as e:
        logger.error(f"Error creating toolbox session: {e}")
        raise

async def delete_session(base_url: str, headers: dict, session_id: str) -> None:
    """Delete a toolbox session using shared HTTP client."""
    try:
        async with http_request('DELETE', f"{base_url}/session/{session_id}", headers=headers, timeout=30) as response:
            if response.status == 204:
                logger.debug(f"Deleted toolbox session: {session_id}")
            else:
                error_text = await response.text()
                logger.warning(f"Failed to delete session {session_id} (status {response.status}): {error_text}")
    except Exception as e:
        logger.error(f"Error deleting toolbox session {session_id}: {e}")

async def main():
    base_url = "https://toolbox-preston.modal-origin.relace.run"
    api_key = "rlc-rtISTILowpUiVcLJjFHUmaYUJ3Y3C9ftGhO-wg"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Test the shared HTTP client with various concurrency levels``
    concurrency_levels = [8, 16, 32, 64, 128, 256, 512]
    
    for N in concurrency_levels:
        print(f"\n{'='*60}")
        print(f"Testing with {N} concurrent requests")
        print(f"{'='*60}")
        
        start_time = asyncio.get_event_loop().time()
        
        # Create N parallel session requests
        tasks = []
        for _ in range(N):
            tasks.append(create_session(base_url, headers))
        
        try:
            # Wait for all sessions to be created
            results = await asyncio.gather(*tasks, return_exceptions=True)
            creation_time = asyncio.get_event_loop().time() - start_time
            
            # Count successful vs failed sessions
            successful_sessions = [r for r in results if isinstance(r, str)]
            failed_sessions = [r for r in results if isinstance(r, Exception)]
            
            print(f"✅ Successfully created: {len(successful_sessions)}/{N} sessions")
            print(f"❌ Failed: {len(failed_sessions)}/{N} sessions")
            print(f"⏱️  Total time: {creation_time:.2f}s")
            
            if failed_sessions:
                print(f"Error samples: {failed_sessions[:3]}")
            
            # Log connection pool stats
            try:
                client = await SharedHTTPClient.get_instance()
                stats = client.get_connection_stats()
                print(f"Connection Pool Stats: {stats['total_connections']}/{stats['max_connections']} connections")
            except Exception as e:
                print(f"Failed to get connection stats: {e}")

            # Wait a bit before cleanup
            await asyncio.sleep(2)

            # Delete successful sessions only
            if successful_sessions:
                delete_start = asyncio.get_event_loop().time()
                delete_tasks = []
                for session_id in successful_sessions:
                    delete_tasks.append(delete_session(base_url, headers, session_id))
                
                await asyncio.gather(*delete_tasks)
                delete_time = asyncio.get_event_loop().time() - delete_start
                print(f"🧹 Cleanup completed in {delete_time:.2f}s")
            else:
                print("🧹 No sessions to cleanup")
            
            # Wait between tests
            await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"❌ Error with {N} concurrent requests: {e}")
            import traceback
            traceback.print_exc()
            break  # Stop testing higher concurrency levels
    
    # Clean up shared HTTP client
    try:
        await SharedHTTPClient.close_instance()
        print("SharedHTTPClient closed successfully")
    except Exception as e:
        print(f"Error closing SharedHTTPClient: {e}")

if __name__ == "__main__":
    asyncio.run(main())
