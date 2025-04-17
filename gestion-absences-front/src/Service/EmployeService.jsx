import axios from 'axios';

const API_URL_EMPLOYE = 'http://localhost:8080/api/employe';


export const getEmployes = async () => {
  const response = await axios.get(API_URL_EMPLOYE);
  return response.data;
};

export const addEmploye = async (employe) => {
  const response = await axios.post(API_URL_EMPLOYE, employe);
  return response.data;
};

export const deleteEmploye = async (id) => {
  await axios.delete(`${API_URL_EMPLOYE}/${id}`);
  return id;
};

export const updateEmploye = async (id, employe) => {
  const response = await axios.put(`${API_URL_EMPLOYE}/${id}`, employe);
  return response.data;
};

export const getDirections = async () => {
    const response = await axios.get(`${API_URL_EMPLOYE}/directions`);
    return response.data;
  };
