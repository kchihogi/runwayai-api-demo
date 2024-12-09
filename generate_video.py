import os
import base64
import asyncio
import requests
from runwayml import AsyncRunwayML
import argparse

client = AsyncRunwayML(
    api_key=os.environ.get("RUNWAYML_API_SECRET"),  # This is the default and can be omitted
)

async def download_file(url, output_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


async def main(first_input_image, second_input_image, prompt_text, output_video_path):
    # check path
    if not os.path.exists(first_input_image):
        print(f"File not found: {first_input_image}")
        return
    
    if not os.path.exists(second_input_image):
        print(f"File not found: {second_input_image}")
        return
    
    # create a parent directory for the output video
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # encode image to base64
    with open(first_input_image, "rb") as f:
        base64_first_image = base64.b64encode(f.read()).decode("utf-8")

    with open(second_input_image, "rb") as f:
        base64_second_image = base64.b64encode(f.read()).decode("utf-8")

    # Create a new image-to-video task using the "gen3a_turbo" model
    task = await client.image_to_video.create(
        model='gen3a_turbo',
        duration=5,  # Duration of the video in seconds
        ratio= "768:1280",  # Aspect ratio of the video. Accepted values:"1280:768" or "768:1280"
        # Point this at your own image file
        prompt_image=[
            {
                "uri": f"data:image/jpeg;base64,{base64_first_image}",
                "position": "first"
            },
            {
                "uri": f"data:image/jpeg;base64,{base64_second_image}",
                "position": "last"
            }
        ],
        prompt_text=prompt_text,
    )

    task_id = task.id

    # Poll the task until it's complete
    await asyncio.sleep(1) 
    task = await client.tasks.retrieve(task_id)
    while task.status not in ['SUCCEEDED', 'FAILED']:
        await asyncio.sleep(1)
        task = await client.tasks.retrieve(task_id)

    print('Task complete:', task)

    video_url = task.output[0]
    await download_file(video_url, output_video_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--first_input_image', type=str, default="data/first.jpg")
    parser.add_argument('--second_input_image', type=str, default="data/second.jpg")
    parser.add_argument('--prompt_text', type=str, default="A blonde-haired woman with straight bangs, wearing a light blue off-shoulder outfit, aiming a gun prop as if about to shoot.")
    parser.add_argument('--output_video_path', type=str, default="data/output/output.mp4")
    args = parser.parse_args()

    first_input_image = args.first_input_image
    second_input_image = args.second_input_image
    prompt_text = args.prompt_text
    output_video_path = args.output_video_path

    print("first_input_image: ", first_input_image)
    print("second_input_image: ", second_input_image)
    print("prompt_text: ", prompt_text)
    print("output_video_path: ", output_video_path)

    asyncio.run(main(first_input_image, second_input_image, prompt_text, output_video_path))

    print("Video generation completed!")
