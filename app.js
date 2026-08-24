const form = document.querySelector('#screeningForm');
const result = document.querySelector('#result');

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(form));
  data.BMI = (Number(data['Weight (Kg)']) / ((Number(data['Height(Cm)']) / 100) ** 2)).toFixed(1);
  const button = form.querySelector('button');
  button.disabled = true;
  button.innerHTML = 'Creating your snapshot…';
  try {
    const response = await fetch('/api/screen', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)});
    if (!response.ok) throw new Error('The local screening service is unavailable.');
    const { probability_pcos } = await response.json();
    const percent = Math.round(probability_pcos * 100);
    const label = percent >= 55 ? 'Worth discussing with a clinician' : 'A lower pattern in this screen';
    result.innerHTML = `<h3>${label}</h3><div class="score">${percent}%</div><p>This is a dataset-based screening estimate—not a diagnosis. PCOS can only be assessed by a qualified clinician who considers your symptoms, history and tests.</p><p>Consider saving your symptoms and cycle history to discuss at your next appointment.</p>`;
    result.hidden = false;
    result.scrollIntoView({behavior: 'smooth', block: 'nearest'});
  } catch (error) {
    result.innerHTML = `<h3>We couldn’t create a snapshot</h3><p>${error.message} Make sure the local server is running, then try again.</p>`;
    result.hidden = false;
  } finally {
    button.disabled = false;
    button.innerHTML = 'View screening snapshot <span>→</span>';
  }
});
