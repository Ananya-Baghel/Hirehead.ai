export default async function handler(req, res) {
  const { resume_text } = req.body;

  const response = await fetch(
    `${process.env.NEXT_PUBLIC_RESUME_API_URL}/analyze`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resume_text })
    }
  );

  const data = await response.json();
  res.status(200).json(data);
}
