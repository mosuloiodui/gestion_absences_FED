import axios from 'axios';

const API_URL_DEMANDE = 'http://localhost:8080/api/demande';


export const getDemandes = async () => {
  const response = await axios.get(API_URL_DEMANDE);
  return response.data;
};

export const addDemande = async (formData) => {
  console.log("axios test");
  const response = await axios.post(API_URL_DEMANDE, formData, 
    {headers: {
      'Content-Type': 'multipart/form-data'
}});
  return response.data;
};

export const deleteDemande = async (id) => {
  await axios.delete(`${API_URL_DEMANDE}/${id}`);
  return id;
};

export const updateDemande = async (id, demande) => {
  const response = await axios.put(`${API_URL_DEMANDE}/${id}`, demande);
  return response.data;
};

export const getMotifs = async () => {
    const response = await axios.get(`${API_URL_DEMANDE}/motif`);
    return response.data;
  };


